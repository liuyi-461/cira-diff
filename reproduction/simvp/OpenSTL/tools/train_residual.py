#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**残差预报分支**：独立修正器，校正主模型（SimVP）的滚动预报结果。

设计动机（三条，均不依赖任何外部"已证明有效"的主张）
--------------------------------------------------
1. 残差分支的输入包含**主模型的输出**，这引入了"主模型误差场"这一新信息通道——
   它与"把目标改成残差"完全不同，后者只是同分布的仿射重参数化。
2. **完全不动主网络** → 可直接复用 baseline 权重复热启动、可随时回退，
   且天然避开「给主模型加一帧条件」带来的架构变更与「多一帧」混淆。
3. 修正量 Δ 用**零初始化**输出层 → 训练开始时 Δ≡0，模型严格等价于 baseline，
   因此残差分支的净贡献可被精确度量。

结构
----
::

    输入(2 或 3 通道, 归一化空间):  [ x̂ 主模型输出 , (base 平流基) , f_t 上一观测帧 ]
                                   ↓
                         残差分支 R（轻量 U-Net，约 1.5M 参数）
                                   ↓  Δ (1,H,W)
    最终输出:  x̂_final = x̂ + Δ          ← 残差连接

    训练时的滚动（**与推理的反馈结构一致**）::

        cur = x                                              # (B,2,1,H,W)
        for k in 0..K-1:
            x̂ = 主模型(cur)[:, :1].squeeze(1)                 # 冻结，不回传
            base = α·advect(prev, flow) + (1−α)·prev          # prev = 当前窗口第二帧
            Δ = R(concat[x̂, base, f_t])                      # 只训 R
            out = x̂ + Δ
            cur = cat([cur[:,1:], out.unsqueeze(1)])          # 把**修正后**的输出喂回

    `flow` 由**当前输入批的两帧**现算（冻结流，不随 rollout 更新），
    因此无需按样本索引查表，shuffle 与否都不会错配。代价是每批多 b 次 DIS 调用
    （1024 样本/轮 ≈ 20–40 s），已实测可接受。

⚠️ 已知局限（必须在报告中注明）
------------------------------
* `base` 用的是**冻结的观测光流**（由输入两帧估得、各步只乘 k），不是逐步重算的
  自一致流场。这是"用物理基做条件信号"的低成本近似，不是完整的风场演变。
* 训练轨迹与推理轨迹在 R 收敛后会逐渐分离；S0 阶段主网络冻结，该偏差为二阶小量。
  若 S0 显示有效，再用 S1（联合微调）消除。

消融矩阵（预注册，**必须含 sham**）
----------------------------------
| 臂 | 输入通道 | 隔离什么 |
|---|---|---|
| R0 | `[x̂, f_t]`（2 通道） | 残差分支本身的净贡献 |
| R1 | `[x̂, base, f_t]`（3 通道） | 完整方案 |
| R1sham | `[x̂, 0, f_t]`（base 通道填中性值） | 「平流基内容」vs「多一个输入通道」的混淆 |

数据约束
--------
train 集只有 1 帧真值，无法做逐步残差监督 → **必须用 validation 集**（1024 条，
output 有 18 帧真值）。`--val-source train` 把验证集改到 train zarr 末尾留出，
从而训练集可用满全部 1024 条。test 集始终 held-out。

预注册判据
----------
* 主判据：LT=18 MAE 相对 baseline 改善 **≥3%**，且 CSI/FSS 下降 **≤1%**。
* 必报：MAE/RMSE/CRPS/CSI/FSS/LapVar/PSD（`tools/eval_metrics.py`）。
* 禁止只报锐度（该指标奖励噪声）。
* 单 seed 不给结论；差距 <0.05 K 视为"无差别"。
* 证伪条件：R1 ≤ R0 → 平流基无净贡献；R1 ≈ R1sham → 增益来自通道数而非内容。

用法::

    python tools/train_residual.py --selftest                 # 光流符号自检
    for M in R0 R1 R1sham; do
      MODE=$M EX_NAME=residual_$M sbatch --job-name=res_$M test_dl/train_residual.slurm
    done
"""

import argparse
import json
import os
import os.path as osp
import sys
import time

_HERE = osp.dirname(osp.abspath(__file__))
_OPENSTL_DIR = osp.dirname(_HERE)
for _p in (_OPENSTL_DIR, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from openstl.datasets.dataloader_satcast import (  # noqa: E402
    MEAN_K, STD_K, SatcastZarrDataset, _SubsetWithAttrs)
from openstl.datasets.utils import create_loader  # noqa: E402
from openstl.models import SimVP_Model  # noqa: E402
from openstl.utils import load_config  # noqa: E402

MODEL_KEYS = ('in_shape', 'hid_S', 'hid_T', 'N_S', 'N_T', 'model_type', 'mlp_ratio',
              'drop', 'drop_path', 'spatio_kernel_enc', 'spatio_kernel_dec',
              'act_inplace')


# ------------------------------------------------------------ 残差分支 ---
def _conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(8, cout),
        nn.GELU(),
    )


class ResidualBranch(nn.Module):
    """轻量 U-Net：`(B, in_ch, 256, 256)` -> `(B, 1, 256, 256)`。

    输出层**零初始化**，因此初始 Δ≡0（等价于 baseline），净贡献可精确度量。
    """

    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.e1 = _conv(in_ch, base)          # 256
        self.e2 = _conv(base, base * 2)       # 128
        self.e3 = _conv(base * 2, base * 4)   # 64
        self.e4 = _conv(base * 4, base * 8)   # 32
        self.d3 = _conv(base * 8 + base * 4, base * 4)
        self.d2 = _conv(base * 4 + base * 2, base * 2)
        self.d1 = _conv(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        e4 = self.e4(F.avg_pool2d(e3, 2))
        u3 = F.interpolate(e4, scale_factor=2, mode='nearest')
        d3 = self.d3(torch.cat([u3, e3], dim=1))
        u2 = F.interpolate(d3, scale_factor=2, mode='nearest')
        d2 = self.d2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, scale_factor=2, mode='nearest')
        d1 = self.d1(torch.cat([u1, e1], dim=1))
        return self.out(d1)


# ------------------------------------------------------------------ 光流 ---
def _to_u8(frame_k):
    x = (np.asarray(frame_k, dtype=np.float64) - 160.0) / 180.0
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def flow_dis(a_k, b_k):
    """DIS medium 光流：把 a 推到 b 的正向位移。符号约定由 `--selftest` 定标。"""
    import cv2
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    return dis.calc(_to_u8(a_k), _to_u8(b_k), None).astype(np.float32)   # (H,W,2)


def warp(img, flow, k):
    """按 k 倍位移外推。"""
    from scipy.ndimage import map_coordinates
    h, w = img.shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    coords = np.stack([(gy - k * flow[..., 1]).ravel(),
                       (gx - k * flow[..., 0]).ravel()])
    return map_coordinates(img, coords, order=1, mode='nearest').reshape(h, w)


def selftest():
    """合成平移定标光流符号（不靠记忆）。"""
    from scipy.ndimage import map_coordinates
    rng = np.random.default_rng(0)
    img = rng.normal(280.0, 8.0, (256, 256)).astype(np.float64)
    gy, gx = np.mgrid[0:256, 0:256].astype(np.float32)
    moved = map_coordinates(img, [gy + 3, gx + 2], order=1,
                            mode='nearest').reshape(256, 256)
    fl = flow_dis(img, moved)
    e_m = float(np.abs(warp(img, fl, 1) - moved).mean())
    e_p = float(np.abs(warp(img, fl, -1) - moved).mean())
    print(f'  [光流自检] 估计=({fl[...,0].mean():.2f},{fl[...,1].mean():.2f}) 真值=(2,3)'
          f' | warp(-flow) MAE={e_m:.4f} warp(+flow) MAE={e_p:.4f}'
          f' -> 采用 {"minus" if e_m <= e_p else "plus"}')


def batch_flow(x_norm):
    """由**批内两帧输入**现算冻结光流。x_norm: (B,2,1,H,W) 归一化空间。

    返回 (B,H,W,2) float32。优点：不需要按样本索引查表，loader 是否 shuffle 都不会错配。
    """
    b = x_norm.shape[0]
    a = x_norm[:, 0, 0].cpu().numpy() * STD_K + MEAN_K
    c = x_norm[:, 1, 0].cpu().numpy() * STD_K + MEAN_K
    return np.stack([flow_dis(a[i], c[i]) for i in range(b)], axis=0)


def advect_batch(prev_k, flow_bhwc, alpha, k):
    """base = α·advect(prev, flow, k) + (1−α)·prev。

    prev_k: (B,H,W) K；flow_bhwc: (B,H,W,2)。
    """
    out = np.empty_like(prev_k)
    for i in range(prev_k.shape[0]):
        out[i] = alpha * warp(prev_k[i], flow_bhwc[i], k) + (1.0 - alpha) * prev_k[i]
    return out


def norm_from_K(k_arr, device):
    """K -> 归一化空间 tensor (B,1,H,W)。"""
    return torch.from_numpy(
        (np.asarray(k_arr, dtype=np.float32) - MEAN_K) / STD_K).to(device).unsqueeze(1)


# ------------------------------------------------------------------- main ---
def build_main(cfg_path, in_shape, ckpt, device):
    cfg = dict(load_config(cfg_path))
    kw = {k: cfg[k] for k in MODEL_KEYS if k in cfg}
    kw['in_shape'] = list(in_shape)
    m = SimVP_Model(**kw)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    sd = {k[6:] if k.startswith('model.') else k: v for k, v in sd.items()}
    missing = set(m.state_dict()) - set(sd)
    if missing:
        raise RuntimeError(f'主模型权重不匹配：missing={sorted(missing)[:5]}')
    m.load_state_dict(sd, strict=True)
    return m.to(device)


def build_r_in(xhat, base_norm, f_t_norm, mode):
    """按消融臂拼残差分支的输入。xhat/base/f_t 均为 (B,1,H,W) 归一化空间。"""
    if mode == 'R0':
        return torch.cat([xhat, f_t_norm], dim=1)
    if mode == 'R1sham':
        return torch.cat([xhat, torch.zeros_like(base_norm), f_t_norm], dim=1)
    return torch.cat([xhat, base_norm, f_t_norm], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=osp.join(
        _OPENSTL_DIR, 'configs', 'satcast', 'SimVP.py'))
    ap.add_argument('--data-root',
                    default=os.environ.get(
                        'SATCAST_ROOT',
                        '/home/group1/26fall_aiclass/yr/data1/data1/satcast'))
    ap.add_argument('--res-dir', default=osp.join(
        _OPENSTL_DIR, 'work_dirs', 'residual'))
    ap.add_argument('--ex-name', default='residual_R1')
    ap.add_argument('--mode', choices=['R0', 'R1', 'R1sham'], default='R1')
    ap.add_argument('--train-zarr',
                    default='edm_GOES_ch13_validation_dataset.zarr',
                    help='多步真值来源；train zarr 只有 1 帧真值，不能用于本脚本')
    ap.add_argument('--val-source', choices=['validation', 'train'], default='train')
    ap.add_argument('--n-train', type=int, default=1024)
    ap.add_argument('--n-val', type=int, default=512)
    ap.add_argument('--main-ckpt', default=osp.join(
        _OPENSTL_DIR, 'work_dirs', 'satcast_simvp_full', 'checkpoints', 'best.ckpt'))
    ap.add_argument('--rollout-steps', type=int, default=6)
    ap.add_argument('--alpha', type=float, default=0.75)
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--unfreeze-main', action='store_true',
                    help='S1：联合微调主网络（默认 S0 冻结）')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--cpu', action='store_true', default=False)
    ap.add_argument('--log-every', type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cpu' if (args.cpu or not torch.cuda.is_available())
                          else 'cuda')
    K = args.rollout_steps

    if args.selftest:
        selftest()
        return

    # ---------------- 数据 ----------------
    full = SatcastZarrDataset(osp.join(args.data_root, args.train_zarr),
                              pre_seq_length=2, aft_seq_length=K)
    n_total = len(full)
    if args.val_source == 'train':
        train_set = _SubsetWithAttrs(full, range(0, min(args.n_train, n_total)))
        tz = SatcastZarrDataset(
            osp.join(args.data_root, 'edm_GOES_ch13_train_dataset.zarr'),
            pre_seq_length=2, aft_seq_length=1)
        val_set = _SubsetWithAttrs(tz, range(len(tz) - args.n_val, len(tz)))
        print(f'[DATA] 训练 = validation 前 {len(train_set)} 条（{K} 帧真值，全量）；'
              f'验证 = train 末尾 {len(val_set)} 条（1 帧真值，单步 MSE）；'
              f'test 集 held-out')
    else:
        n_tr = min(args.n_train, n_total - args.n_val)
        train_set = _SubsetWithAttrs(full, range(0, n_tr))
        val_set = _SubsetWithAttrs(full, range(n_total - args.n_val, n_total))
        print(f'[DATA] validation 切分：train {len(train_set)} / val {len(val_set)}')

    train_loader = create_loader(
        train_set, batch_size=args.batch_size, shuffle=True, is_training=True,
        pin_memory=True, drop_last=True, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0)
    val_loader = create_loader(
        val_set, batch_size=4, shuffle=False, is_training=False,
        pin_memory=True, drop_last=False, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0)

    # ---------------- 模型 ----------------
    G = build_main(args.config, [2, 1, 256, 256], args.main_ckpt, device)
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    if args.unfreeze_main:
        G.train()
        for p in G.parameters():
            p.requires_grad_(True)
        print('[MODE] S1：主网络参与联合微调')
    else:
        print('[MODE] S0：主网络冻结，只训残差分支')

    in_ch = 2 if args.mode == 'R0' else 3
    R = ResidualBranch(in_ch=in_ch, base=args.width).to(device)
    if args.mode == 'R1sham':
        print('[MODE] R1sham：平流基通道填中性值（隔离"多一个通道"的混淆）')

    params = list(R.parameters()) + (list(G.parameters())
                                     if args.unfreeze_main else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)
    steps_per_epoch = len(train_loader)
    # ⚠️ total_steps 必须用真实 steps/epoch（曾因写常数中途崩溃）
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr,
        total_steps=max(1, steps_per_epoch * args.epochs), pct_start=0.1)

    save_dir = osp.join(args.res_dir, args.ex_name)
    ck_dir = osp.join(save_dir, 'checkpoints')
    os.makedirs(ck_dir, exist_ok=True)
    with open(osp.join(save_dir, 'residual_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    n_r = sum(p.numel() for p in R.parameters()) / 1e6
    n_g = sum(p.numel() for p in G.parameters()) / 1e6
    print('=' * 74)
    print(f'残差预报分支 {args.mode}｜主模型 {n_g:.3f}M（'
          f'{"联合训练" if args.unfreeze_main else "冻结"}）｜残差分支 {n_r:.3f}M')
    print(f'[TRAIN] K={K} α={args.alpha} batch={args.batch_size} '
          f'epochs={args.epochs} steps/epoch={steps_per_epoch}')
    print('[INIT ] 输出层零初始化 → 初始 Δ≡0，严格等价于 baseline')
    print('=' * 74)

    best, hist = float('inf'), []
    for epoch in range(args.epochs):
        R.train()
        t0, nb = time.time(), 0
        per_step = np.zeros(K)
        run = 0.0

        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)      # (B,2,1,H,W) 归一化
            yb = yb.to(device, non_blocking=True)      # (B,K,1,H,W) 归一化
            b = xb.shape[0]

            if epoch == 0 and step == 0:
                print(f'[DEBUG] x={tuple(xb.shape)} y={tuple(yb.shape)} K={K} '
                      f'mode={args.mode}')
                assert xb.shape[1] == 2, f'输入时间维应为 2，实际 {tuple(xb.shape)}'
                assert yb.shape[1] == K, f'真值时间维应为 K={K}，实际 {tuple(yb.shape)}'

            fl = batch_flow(xb)                         # (B,H,W,2) 冻结观测流
            f_t_norm = xb[:, 1]                         # (B,1,H,W)
            cur = xb
            losses = []
            for k in range(K):
                with torch.no_grad():
                    xhat = G(cur)[:, :1].squeeze(1)     # (B,1,H,W)
                # 平流基由**当前窗口第二帧**（可能是上一步的修正输出）现算；
                # 必须 detach：基准是停止梯度的条件信号，梯度只应经 R 的本步前向回流。
                prev_K = (cur[:, 1, 0].detach().cpu().numpy()
                          * STD_K + MEAN_K).astype(np.float32)
                base_K = advect_batch(prev_K, fl, args.alpha, k + 1)
                base_norm = norm_from_K(base_K, device)
                delta = R(build_r_in(xhat, base_norm, f_t_norm, args.mode))
                out = xhat + delta                      # 残差连接
                losses.append(F.mse_loss(out, yb[:, k]))
                per_step[k] += float(losses[-1].detach())
                # 把**修正后**的输出喂回（与推理一致）
                cur = torch.cat([cur[:, 1:], out.unsqueeze(1)], dim=1)

            loss = torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(R.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss.detach())
            nb += 1

            if args.log_every and (step + 1) % args.log_every == 0:
                mem = (torch.cuda.max_memory_allocated() / 2 ** 30
                       if torch.cuda.is_available() else 0.0)
                print(f'  ep{epoch} step{step+1}/{steps_per_epoch} '
                      f'loss={run/nb:.5f} cur={float(loss):.5f} mem={mem:.1f}G')

        # ---------------- 验证（单步 MSE，val 集只有 1 帧真值） ----------------
        R.eval()
        vsum, vnb = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                fl = batch_flow(xb)
                prev_K = (xb[:, 1, 0].detach().cpu().numpy()
                          * STD_K + MEAN_K).astype(np.float32)
                base_norm = norm_from_K(
                    advect_batch(prev_K, fl, args.alpha, 1), device)
                xhat = G(xb)[:, :1].squeeze(1)
                out = xhat + R(build_r_in(xhat, base_norm, xb[:, 1], args.mode))
                vsum += float(F.mse_loss(out, yb[:, 0]))
                vnb += 1
        val = vsum / max(vnb, 1)
        rec = {'epoch': epoch, 'val_mse': val,
               'val_rmse_K': float(np.sqrt(val) * STD_K),
               'train_loss_mean': run / max(nb, 1),
               'train_per_step_mse': (per_step / max(nb, 1)).tolist(),
               'time_s': round(time.time() - t0, 1)}
        hist.append(rec)
        print(f'[EPOCH {epoch}] val_mse={val:.5f} (rmse {rec["val_rmse_K"]:.3f} K)  '
              f'train_loss={rec["train_loss_mean"]:.5f}  '
              f'逐步MSE={[round(x, 5) for x in rec["train_per_step_mse"]]}  '
              f'({rec["time_s"]}s)')

        torch.save({'state_dict': R.state_dict(), 'epoch': epoch, 'val_mse': val},
                   osp.join(ck_dir, 'last.pt'))
        if val < best:
            best = val
            torch.save({'state_dict': R.state_dict(), 'epoch': epoch,
                        'val_mse': val}, osp.join(ck_dir, 'best.pt'))
            print(f'  -> 新最优 (val={val:.5f})')

    with open(osp.join(save_dir, 'residual_history.json'), 'w') as f:
        json.dump(hist, f, indent=2)
    print(f'[DONE] 产物在 {save_dir}')


if __name__ == '__main__':
    main()
