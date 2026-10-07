#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SimVP 的 **baseline 对齐评估**（统一协议 v1，见 docs/experiments/EXP-004）。

为什么要单独写这个脚本（而不是用 OpenSTL 的 ``exp.test()``）:

  1. OpenSTL 的 ``trainer.test()`` 跑在**最终权重**上，而 baseline 要求用 **best.ckpt**。
  2. ``openstl/core/metrics.py`` 的 MSE/MAE/RMSE 对空间维用 ``.sum()``，会把
     MSE/MAE 放大 65536 倍、RMSE 放大 256 倍（见 EXP-004 "口径坑记录"）。
     为遵守 ``DEC-003``（不改上游），这里**不修改**上游，而是自己按 baseline
     的**逐像素**口径计算，与 ly 的 ``F.mse_loss`` 口径一致。
  3. baseline（ly 的 UNet）报的是单步 + 18 步 rollout、逐 LT 统计；本脚本复刻同一协议，
     并同时给出**归一化空间**与**亮温(K)**两套数字。

数值空间
--------
zarr 内已是归一化空间（mean=0, std=1）。换算：``Tb(K) = zarr * STD_K + MEAN_K``。
误差类指标：``MAE_K = MAE_norm * STD_K``、``MSE_K2 = MSE_norm * STD_K**2``、
``RMSE_K = RMSE_norm * STD_K``。

老 checkpoint 的归一化还原
--------------------------
EXP-002 的 smoke 模型是用**子集自算**的 ``mean=0.093006 / std=0.865920`` 训练的，
不是物理空间。评估它时必须先用同一组常数把输入还原到它训练时的空间，
再把输出逆变换回 zarr 空间：

    xn    = (zarr - norm_mean) / norm_std
    pred  = model(xn) * norm_std + norm_mean      # 回到 zarr 空间

即 ``--norm-mean 0.093006 --norm-std 0.865920``。
按 baseline 协议新训的模型（直接吃 zarr 原值）用默认值 0 / 1 即可。

用法
----
    cd <OpenSTL>
    python tools/eval_simvp_baseline.py \
        --ckpt work_dirs/simvp_goes13_smoke/checkpoints/best.ckpt \
        --config configs/goes13/simvp/SimVP_gSTA.py \
        --norm-mean 0.093006 --norm-std 0.865920

    # 3 小时 rollout（对齐 ly 的 LT=1..18）
    python tools/eval_simvp_baseline.py --ckpt <...>/best.ckpt --rollout-steps 18
"""

import argparse
import json
import math
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = osp.dirname(osp.abspath(__file__))
OPENSTL_DIR = osp.dirname(HERE)
if OPENSTL_DIR not in sys.path:
    sys.path.insert(0, OPENSTL_DIR)

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from openstl.datasets.dataloader_satcast import (  # noqa: E402
    DEFAULT_TEST_ZARR, MEAN_K, STD_K, SatcastZarrDataset)
from openstl.models import SimVP_Model  # noqa: E402
from openstl.utils import load_config  # noqa: E402

# SimVP_Model.__init__ 接受的键（config 里还有 lr/epoch 等训练超参，要过滤掉）
MODEL_KEYS = ('in_shape', 'hid_S', 'hid_T', 'N_S', 'N_T', 'model_type', 'mlp_ratio',
              'drop', 'drop_path', 'spatio_kernel_enc', 'spatio_kernel_dec',
              'act_inplace')

DEFAULT_DATA_ROOT = '/home/group1/26fall_aiclass/yr/data1/data1/satcast'


# --------------------------------------------------------------- metrics ---
def ssim_batch(p, t, win=11, c1=0.01 ** 2, c2=0.03 ** 2):
    """逐样本 SSIM，口径与 ly 的评估脚本一致（11x11 窗口，avg_pool2d）。

    Args:
        p, t: (N, 1, H, W) float32 tensor
    Returns:
        (N,) 每样本的 SSIM（在空间上取均值）
    """
    pad = win // 2

    def filt(x):
        return F.avg_pool2d(x, win, stride=1, padding=pad)

    mu_p, mu_t = filt(p), filt(t)
    mu_pp, mu_tt, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    sig_p = filt(p * p) - mu_pp
    sig_t = filt(t * t) - mu_tt
    sig_pt = filt(p * t) - mu_pt
    num = (2.0 * mu_pt + c1) * (2.0 * sig_pt + c2)
    den = (mu_pp + mu_tt + c1) * (sig_p + sig_t + c2)
    return (num / den).mean(dim=(1, 2, 3))


def per_sample_metrics(p, t):
    """逐样本指标（归一化空间）。p, t: (N,1,H,W) tensor。

    口径：逐像素（对全部元素求均值），与 ly 的 F.mse_loss / F.l1_loss 一致。
    注意：不是 OpenSTL metrics.py 的 .sum() 口径。
    """
    mse = ((p - t) ** 2).mean(dim=(1, 2, 3))
    mae = (p - t).abs().mean(dim=(1, 2, 3))
    rmse = torch.sqrt(mse)
    ssim = ssim_batch(p, t)
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-12))
    return {
        'mse': mse.cpu().numpy(),
        'mae': mae.cpu().numpy(),
        'rmse': rmse.cpu().numpy(),
        'ssim': ssim.cpu().numpy(),
        'psnr': psnr.cpu().numpy(),
    }


def summarize(values):
    """把逐样本指标数组汇总成 mean/std/median（对齐 ly 的报告方式）。"""
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
    }


def summarize_metrics(per):
    """汇总逐样本指标，并**显式区分 RMSE 的两种口径**。

    ⚠️ RMSE 口径陷阱（Jensen 不等式）：
        * ``rmse_global``        = sqrt(全局 MSE)          —— cb 用的口径
        * ``rmse_persample_mean``= mean(逐样本 RMSE)       —— ly 的"逐样本再取均值"口径
      两者**不相等**，且 E[√X] ≤ √(E[X])，后者系统性偏小。
      实测同一份 persistence：global=4.5492 K，per-sample mean=3.8772 K（差 15%）。

      MSE / MAE 无此歧义：样本像素数相同，mean(逐样本 MSE) == 全局 MSE。
      因此**横向对比 cb 必须用 rmse_global**；本函数两个都给，避免误用。
    """
    out = {k: summarize(v) for k, v in per.items()}
    out['rmse_global'] = float(math.sqrt(out['mse']['mean']))
    out['rmse_persample_mean'] = out['rmse']['mean']
    return out


def to_K(m):
    """归一化空间 -> 亮温 K（误差类指标乘 STD_K；SSIM/PSNR 无量纲不换算）。"""
    # 显式转 python float：numpy.float32 无法被 json 序列化
    return {
        'mse_K2': float(m['mse']['mean'] * STD_K ** 2),
        'mae_K': float(m['mae']['mean'] * STD_K),
        # 主指标用 global 口径，才能与 cb 的 rmse_K 直接比较
        'rmse_K': float(m['rmse_global'] * STD_K),
        'rmse_K_persample_mean': float(m['rmse_persample_mean'] * STD_K),
    }


# ----------------------------------------------------------------- model ---
def build_model(config_path, in_shape):
    cfg = dict(load_config(config_path))
    kwargs = {k: cfg[k] for k in MODEL_KEYS if k in cfg}
    kwargs['in_shape'] = list(in_shape)
    return SimVP_Model(**kwargs), kwargs


def load_weights(model, ckpt_path):
    """把 Lightning checkpoint 的权重装进裸 SimVP_Model（剥 'model.' 前缀并严格校验）。"""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    sd = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd.items()}
    model_keys = set(model.state_dict().keys())
    missing = model_keys - set(sd.keys())
    unexpected = set(sd.keys()) - model_keys
    if missing or unexpected:
        raise RuntimeError(
            f'权重与模型不匹配：missing={sorted(missing)[:5]} '
            f'unexpected={sorted(unexpected)[:5]}')
    model.load_state_dict(sd, strict=True)
    return ckpt


# ------------------------------------------------------------- inference ---
@torch.no_grad()
def run(model, dataset, device, batch_size, norm_mean, norm_std, steps, max_samples):
    """单步 + 可选 rollout 推理。

    Returns:
        pred:  (steps, N, 1, H, W)  zarr 空间的预测
        true:  (steps, N, 1, H, W)  zarr 空间的真值
        persist:(steps, N, 1, H, W) persistence 基线（最近输入帧，rollout 时保持冻结）
    """
    model.eval()
    n = len(dataset)
    if max_samples and max_samples > 0:
        n = min(n, max_samples)

    preds, trues, persists = [], [], []
    for start in range(0, n, batch_size):
        idxs = list(range(start, min(start + batch_size, n)))
        xb = torch.stack([dataset[i][0] for i in idxs]).to(device)   # (b,2,1,H,W) zarr空间
        yb = torch.stack([dataset[i][1] for i in idxs]).to(device)   # (b,steps,1,H,W)

        # 还原到模型训练时的空间
        xn = (xb - norm_mean) / norm_std

        cur = xn
        step_preds = []
        for _ in range(steps):
            # SimVP 在 aft<pre 时输出 pre_seq_length 帧，取前 1 帧即为下一帧；
            # squeeze(1) 去掉长度为 1 的时间维 -> (b, C, H, W)
            p = model(cur)[:, :1].squeeze(1)
            step_preds.append(p)
            cur = torch.cat([cur[:, 1:], p.unsqueeze(1)], dim=1)
        # (steps, b, C, H, W) -> 逆变换回 zarr 空间
        pred = torch.stack(step_preds, dim=0) * norm_std + norm_mean

        preds.append(pred.cpu())
        trues.append(yb.permute(1, 0, 2, 3, 4).cpu())          # (steps, b, C, H, W)
        # persistence：最近输入帧 t 当作 t+10min；rollout 时冻结该帧
        persist_frame = xb[:, -1:].squeeze(1)                  # (b, C, H, W)
        persists.append(
            persist_frame.unsqueeze(0).expand(steps, -1, -1, -1, -1).cpu())

    cat = lambda xs: torch.cat(xs, dim=1)  # noqa: E731
    return cat(preds), cat(trues), cat(persists)


# ------------------------------------------------------------------ plots ---
def save_curves(out_png, per_lt):
    lts = sorted(per_lt.keys(), key=int)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, key, title in (
            (axes[0], 'mse', 'MSE vs lead time (normalized space)'),
            (axes[1], 'ssim', 'SSIM vs lead time')):
        ax.plot(lts, [per_lt[lt]['simvp'][key]['mean'] for lt in lts],
                'o-', label='SimVP')
        ax.plot(lts, [per_lt[lt]['persistence'][key]['mean'] for lt in lts],
                's--', label='persistence')
        ax.set_xlabel('lead time (steps, 1 step = 10 min)')
        ax.set_ylabel(key.upper())
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def save_panels(out_png, pred, true, persist, n_show=3):
    n_show = min(n_show, pred.shape[1])
    fig, axes = plt.subplots(n_show, 5, figsize=(22, 4.2 * n_show))
    axes = np.atleast_2d(axes)
    for r in range(n_show):
        gt, pd, ps = true[0, r, 0], pred[0, r, 0], persist[0, r, 0]
        panels = (
            (gt, f'#{r} Target t+10min', 'Spectral_r', -4, 2),
            (pd, f'#{r} SimVP pred', 'Spectral_r', -4, 2),
            (ps, f'#{r} Persistence', 'Spectral_r', -4, 2),
            (np.abs(pd - gt), f'#{r} |SimVP - GT|', 'hot', None, None),
            (np.abs(ps - gt), f'#{r} |Persistence - GT|', 'hot', None, None),
        )
        for ax, (img, title, cmap, vmin, vmax) in zip(axes[r], panels):
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------- main ---
def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    steps = int(args.rollout_steps)
    in_shape = [int(v) for v in args.in_shape.split(',')]

    model, kwargs = build_model(args.config, in_shape)
    ckpt = load_weights(model, args.ckpt)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    zarr_path = args.zarr if osp.isabs(args.zarr) else osp.join(args.data_root, args.zarr)
    dataset = SatcastZarrDataset(zarr_path, pre_seq_length=in_shape[0],
                                 aft_seq_length=steps)

    # 样本量：对齐 ly 协议（单步用全量 1024；rollout 用 128 子集）
    max_samples = args.max_samples
    if max_samples is None or int(max_samples) <= 0:
        max_samples = 128 if steps > 1 else 0

    print('=' * 78)
    print('SimVP baseline 对齐评估（统一协议 v1）')
    print('=' * 78)
    print(f'[ENV ] device      : {device}')
    print(f'[CKPT] {args.ckpt}')
    print(f'[CKPT] best epoch  : {ckpt.get("epoch", "N/A")}')
    print(f'[MODEL] {kwargs}')
    print(f'[MODEL] 参数量     : {n_params / 1e6:.3f} M')
    print(f'[DATA] {zarr_path}  样本={len(dataset)}  本轮用='
          f'{min(len(dataset), max_samples) if max_samples else len(dataset)}')
    print(f'[DATA] output 原始帧数 {dataset.out_frames}，本任务取前 {steps} 帧 '
          f'(output[:,0] = t+10min)')
    print(f'[NORM] 还原常数     : mean={args.norm_mean} std={args.norm_std} '
          f'(0/1 表示直接吃 zarr 原值)')
    print(f'[PROTO] rollout={steps} 步（{steps * 10} 分钟）  batch_size={args.batch_size}')

    pred, true, persist = run(model, dataset, device, args.batch_size,
                              args.norm_mean, args.norm_std, steps, max_samples)
    assert pred.shape == true.shape, (pred.shape, true.shape)

    per_lt = {}
    for lt in range(steps):
        p, t, ps = pred[lt], true[lt], persist[lt]
        ms = summarize_metrics(per_sample_metrics(p, t))
        mp = summarize_metrics(per_sample_metrics(ps, t))
        per_lt[str(lt + 1)] = {
            'simvp': ms,
            'simvp_K': to_K(ms),
            'persistence': mp,
            'persistence_K': to_K(mp),
        }

    print('\n' + '-' * 78)
    head = f'{"LT":>3}  {"MSE(SimVP)":>12} {"RMSE(K)":>10} {"MAE(K)":>9} {"SSIM":>7}' \
           f' | {"RMSE(K,persist)":>16}'
    print(head)
    print('-' * 78)
    for lt in range(steps):
        d = per_lt[str(lt + 1)]
        print(f'{lt + 1:>3}  {d["simvp"]["mse"]["mean"]:>12.6f} '
              f'{d["simvp_K"]["rmse_K"]:>10.4f} {d["simvp_K"]["mae_K"]:>9.4f} '
              f'{d["simvp"]["ssim"]["mean"]:>7.4f} | '
              f'{d["persistence_K"]["rmse_K"]:>16.4f}')

    first = per_lt['1']
    red = (1.0 - first['simvp']['mse']['mean'] / first['persistence']['mse']['mean']) * 100
    print('-' * 78)
    print(f'单步(LT=1) SimVP       : MSE={first["simvp"]["mse"]["mean"]:.6f} '
          f'RMSE={first["simvp_K"]["rmse_K"]:.4f} K  '
          f'MAE={first["simvp_K"]["mae_K"]:.4f} K  '
          f'SSIM={first["simvp"]["ssim"]["mean"]:.4f}')
    print(f'单步(LT=1) persistence : MSE={first["persistence"]["mse"]["mean"]:.6f} '
          f'RMSE={first["persistence_K"]["rmse_K"]:.4f} K')
    print(f'MSE 相对 persistence 降低：{red:.2f}%')
    print('[口径] RMSE 主值 = sqrt(全局MSE)，与 cb 可比；'
          '逐样本均值口径（与 ly 可比）供参考：'
          f'SimVP {first["simvp_K"]["rmse_K_persample_mean"]:.4f} K / '
          f'persist {first["persistence_K"]["rmse_K_persample_mean"]:.4f} K')
    print('\n[baseline 参照] UNet 单步 MSE≈0.00809 (RMSE≈1.74 K)；'
          'SimVP(cb 全量) RMSE≈1.57 K；persistence≈4.55 K')

    os.makedirs(args.out_dir, exist_ok=True)
    curve_png = osp.join(args.out_dir, 'baseline_curves.png')
    panel_png = osp.join(args.out_dir, 'baseline_panels.png')
    try:
        save_curves(curve_png, per_lt)
        save_panels(panel_png, pred, true, persist)
        print(f'[SAVE] 曲线图 : {curve_png}')
        print(f'[SAVE] 对比图 : {panel_png}')
    except Exception as exc:
        print(f'[WARNING] 出图失败（不影响指标）：{type(exc).__name__}: {exc}')

    result = {
        'protocol': 'baseline-v1 (per-pixel; test zarr; best.ckpt; 1 step = 10 min)',
        'ckpt': args.ckpt,
        'ckpt_epoch': ckpt.get('epoch', None),
        'config': args.config,
        'model_kwargs': {k: v for k, v in kwargs.items() if k != 'in_shape'},
        'in_shape': in_shape,
        'params_M': round(n_params / 1e6, 4),
        'zarr': zarr_path,
        'n_samples': int(pred.shape[1]),
        'rollout_steps': steps,
        'norm_restore': {'mean': args.norm_mean, 'std': args.norm_std},
        'physical': {'MEAN_K': MEAN_K, 'STD_K': STD_K},
        'per_lead_time': per_lt,
        'mse_reduction_vs_persistence_pct_LT1': red,
    }
    out_json = osp.join(args.out_dir, 'baseline_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'[SAVE] 指标 JSON : {out_json}')

    np.save(osp.join(args.out_dir, 'preds.npy'), pred.numpy().astype(np.float32))
    np.save(osp.join(args.out_dir, 'trues.npy'), true.numpy().astype(np.float32))
    np.save(osp.join(args.out_dir, 'persist.npy'), persist.numpy().astype(np.float32))
    print(f'[SAVE] preds/trues/persist .npy -> {args.out_dir}')
    return result


def main():
    ap = argparse.ArgumentParser(
        description='SimVP baseline 对齐评估（统一协议 v1）')
    ap.add_argument('--ckpt', required=True, help='best.ckpt 路径')
    ap.add_argument('--config', default=osp.join(
        OPENSTL_DIR, 'configs', 'goes13', 'simvp', 'SimVP_gSTA.py'))
    ap.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--zarr', default=DEFAULT_TEST_ZARR)
    ap.add_argument('--in_shape', default='2,1,256,256')
    ap.add_argument('--out_dir', default=osp.join(
        OPENSTL_DIR, 'work_dirs', 'baseline_eval'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--rollout-steps', type=int, default=1,
                    help='1 = 仅单步；18 = 对齐 ly 的 3 小时 rollout')
    ap.add_argument('--max-samples', type=int, default=0,
                    help='0/负数=自动（单步全量、rollout 用 128，对齐 ly 协议）')
    ap.add_argument('--norm-mean', type=float, default=0.0,
                    help='模型训练时用的归一化 mean（老 smoke ckpt 用 0.093006）')
    ap.add_argument('--norm-std', type=float, default=1.0,
                    help='模型训练时用的归一化 std（老 smoke ckpt 用 0.865920）')
    args = ap.parse_args()

    if not osp.exists(args.ckpt):
        raise FileNotFoundError(f'找不到 checkpoint：{args.ckpt}')
    evaluate(args)


if __name__ == '__main__':
    main()
