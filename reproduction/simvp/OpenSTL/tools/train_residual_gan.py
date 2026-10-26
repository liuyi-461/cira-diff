#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**SimVP + 残差分支 + 判别器** 联合训练：残差与 GAN、SimVP 一起做云图预报。

结构
----
::

    输入 x (2 帧，归一化空间)
       │
       ▼
    ┌──────────────────────────────┐
    │  SimVP G（默认冻结，可解冻）  │ -> x̂  主预报（确定性基）
    └──────────────────────────────┘
       │                            │
       │  base = α·advect(prev,flow)+(1−α)·prev   （物理平流基，条件信号）
       │                            │
       ▼                            ▼
    ┌──────────────────────────────────────┐
    │ 残差分支 R([x̂, base, f_t]) -> Δ       │  轻量 U-Net，输出层零初始化
    └──────────────────────────────────────┘
       │
       ▼
    out = x̂ + Δ                        ← 最终预报（残差连接）
       │
       ├─ MSE：mean_k MSE(out_k, y_k)                  （全部 K 步）
       ├─ rel_grad：双向锐度正则（作用于**整场输出**）
       └─ 对抗：D 判别**残差场 Δ**（不是整场！）

    判别器输入：
      真实残差 = y_k − x̂_k     （"本该做的修正"）
      假残差   = Δ_k = R(...)   （"实际做的修正"）

为什么判别器只压残差（三条依据）
------------------------------
1. **LAPGAN / MelanoGANs 先例**：多级金字塔里每一级生成器负责生成**残差图**
   （高频成分），判别器判别的就是该级残差，而非最终整图。
2. **本项目的反面实证**：此前把判别器压在整场上（`train_simvp_gan.py`），
   锐度比冲到 2.969、RMSE 恶化 34.8% —— 判别器会去"修"本已由 SimVP +
   平流基保证的大尺度位移结构，既浪费容量又制造假纹理。
3. **残差场统计特性更集中**：Δ 里基本只剩小尺度生消（云边界、对流单体生消），
   本就应该是尖锐的，判别器只需建模这一小块高频分布，容易收敛。

理论依据：残差/增量学习（Δ-ML）
--------------------------------
    y_target = y_baseline + Δ_baseline→target
模型只学"廉价基线"与"高精目标"之间的差，兼具前者速度与后者精度。
残差分支的输入含**主模型的输出**，因此引入了"主模型误差场"这一新信息通道——
这与"把目标改成残差"完全不同（后者只是同分布的仿射重参数化）。

⚠️ 与 SS 的关系
--------------
本脚本**不使用** scheduled sampling，也不与任何 SS 实验对比
（用户明确要求）。滚动时条件帧一律用模型自身预测（标准自回归）。

数据约束
--------
train 集 `output` 仅 1 帧真值，逐步监督不成立 → 必须用 **validation 集**
（18 帧真值）。`--val-source train` 把验证集改到 train zarr 末尾留出，
从而训练集可用满全部 1024 条。test 集始终 held-out。

预注册判据
----------
* 主判据：LT=18 MAE 相对 **UNet（official）** 改善，且 CSI/FSS 不下降超过 1%。
* 必报：`tools/eval_metrics.py` 的 8 指标（MAE/RMSE/CRPS/SSIM/CSI/FSS/PSD/LapVar）。
* 禁止只报锐度（该指标奖励噪声）。
* 消融：`--mode R0`（无平流基）/ `R1`（含平流基）/ `R1sham`（平流基填中性值），
  用于隔离"平流基内容"与"多一个输入通道"的混淆。

用法::

    python tools/train_residual_gan.py --selftest
    MODE=R1 GAN_WEIGHT=0.05 sbatch --job-name=resgan_R1 test_dl/train_residual_gan.slurm
    # 出预测（供 eval_metrics / 云图对比）
    python tools/make_residual_preds.py --ex-name resgan_R1 --ckpt <best.pt>
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
import torch.nn.functional as F  # noqa: E402

from openstl.datasets.dataloader_satcast import (  # noqa: E402
    MEAN_K, STD_K, SatcastZarrDataset, _SubsetWithAttrs)
from openstl.datasets.utils import create_loader  # noqa: E402

# 复用残差分支与判别器的既有实现，避免复制代码
from train_residual import (ResidualBranch, advect_batch, batch_flow,  # noqa: E402
                            build_main, build_r_in, norm_from_K,
                            selftest)
from train_simvp_gan import (PatchDiscriminator, grad_magnitude,  # noqa: E402
                             rel_grad_loss)


# ------------------------------------------------------------------ 滚动 ---
def rollout_residual_gan(main, R, xb, K, alpha, mode, device, flow=None):
    """滚动 K 步：主模型出基预报 → 残差修正 → 把**修正后**输出喂回。

    Returns:
        outs:   (K,B,1,H,W) 最终预报 x̂+Δ（带梯度，经 Δ）
        deltas: (K,B,1,H,W) 残差修正量 Δ
        xhats:  (K,B,1,H,W) 主模型基预报 x̂（已 detach）
    """
    fl = batch_flow(xb) if flow is None else flow      # (B,H,W,2) 冻结观测流
    f_t_norm = xb[:, 1]                                # (B,1,H,W)
    cur = xb
    outs, deltas, xhats = [], [], []
    for k in range(K):
        with torch.no_grad():
            xhat = main(cur)[:, :1].squeeze(1)         # (B,1,H,W)
        # 平流基由当前窗口第二帧现算；detach：基准是停止梯度的条件信号
        prev_K = (cur[:, 1, 0].detach().cpu().numpy()
                  * STD_K + MEAN_K).astype(np.float32)
        base_norm = norm_from_K(advect_batch(prev_K, fl, alpha, k + 1), device)
        delta = R(build_r_in(xhat, base_norm, f_t_norm, mode))
        out = xhat + delta                             # 残差连接
        outs.append(out)
        deltas.append(delta)
        xhats.append(xhat)
        cur = torch.cat([cur[:, 1:], out.unsqueeze(1)], dim=1)
    return (torch.stack(outs), torch.stack(deltas), torch.stack(xhats))


# ------------------------------------------------------------------- main ---
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
    ap.add_argument('--ex-name', default='resgan_R1')
    ap.add_argument('--mode', choices=['R0', 'R1', 'R1sham'], default='R1')
    ap.add_argument('--train-zarr',
                    default='edm_GOES_ch13_validation_dataset.zarr')
    ap.add_argument('--val-source', choices=['validation', 'train'],
                    default='train')
    ap.add_argument('--n-train', type=int, default=1024)
    ap.add_argument('--n-val', type=int, default=512)
    ap.add_argument('--main-ckpt', default=osp.join(
        _OPENSTL_DIR, 'work_dirs', 'satcast_simvp_full', 'checkpoints', 'best.ckpt'))
    ap.add_argument('--rollout-steps', type=int, default=6)
    ap.add_argument('--alpha', type=float, default=0.75)
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3, help='残差分支学习率')
    ap.add_argument('--lr-d', type=float, default=1e-4, help='判别器学习率')
    ap.add_argument('--gan-weight', type=float, default=0.05,
                    help='对抗损失权重，**相对 MSE 的比例**'
                         '（0.05 = 约占 MSE 的 5%%）')
    ap.add_argument('--grad-weight', type=float, default=0.05,
                    help='双向锐度正则权重，**相对 MSE 的比例**')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--unfreeze-main', action='store_true',
                    help='S1：联合微调 SimVP 主干（默认 S0 冻结）')
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
        print(f'[DATA] 训练 = validation 前 {len(train_set)} 条（{K} 帧真值）；'
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
        print('[MODE] S1：SimVP 主干参与联合微调')
    else:
        print('[MODE] S0：SimVP 主干冻结，只训残差分支 + 判别器')

    in_ch = 2 if args.mode == 'R0' else 3
    R = ResidualBranch(in_ch=in_ch, base=args.width).to(device)
    D = PatchDiscriminator(in_ch=1, base=64, n_layers=4).to(device)
    if args.mode == 'R1sham':
        print('[MODE] R1sham：平流基通道填中性值（隔离"多一个通道"的混淆）')

    params = list(R.parameters()) + (list(G.parameters())
                                     if args.unfreeze_main else [])
    opt_g = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)
    opt_d = torch.optim.AdamW(D.parameters(), lr=args.lr_d, weight_decay=0)
    steps_per_epoch = len(train_loader)
    # ⚠️ total_steps 必须用真实 steps/epoch（曾因写常数中途崩溃）
    sched_g = torch.optim.lr_scheduler.OneCycleLR(
        opt_g, max_lr=args.lr,
        total_steps=max(1, steps_per_epoch * args.epochs), pct_start=0.1)

    save_dir = osp.join(args.res_dir, args.ex_name)
    ck_dir = osp.join(save_dir, 'checkpoints')
    os.makedirs(ck_dir, exist_ok=True)
    with open(osp.join(save_dir, 'resgan_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    n_r = sum(p.numel() for p in R.parameters()) / 1e6
    n_d = sum(p.numel() for p in D.parameters()) / 1e6
    n_g = sum(p.numel() for p in G.parameters()) / 1e6
    print('=' * 76)
    print(f'SimVP({n_g:.3f}M,{"联合" if args.unfreeze_main else "冻结"}) + '
          f'残差分支({n_r:.3f}M) + 判别器({n_d:.3f}M)')
    print(f'[TRAIN] K={K} α={args.alpha} batch={args.batch_size} '
          f'epochs={args.epochs} steps/epoch={steps_per_epoch}')
    print(f'[LOSS ] mse × (1 + {args.gan_weight}·adv + {args.grad_weight}·rel_grad)')
    print('[DISC ] 判别器作用于**残差场 Δ**（真实残差=y−x̂，假残差=R 输出）')
    print('[INIT ] 残差输出层零初始化 → 初始 Δ≡0，严格等价于 baseline')
    print('=' * 76)

    best, hist = float('inf'), []
    for epoch in range(args.epochs):
        R.train()
        D.train()
        t0, nb = time.time(), 0
        per_step = np.zeros(K)
        acc = {'mse': 0.0, 'adv': 0.0, 'dloss': 0.0, 'rg': 0.0,
               'gm_real': 0.0, 'gm_delta': 0.0}

        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)      # (B,2,1,H,W)
            yb = yb.to(device, non_blocking=True)      # (B,K,1,H,W)
            b = xb.shape[0]

            if epoch == 0 and step == 0:
                print(f'[DEBUG] x={tuple(xb.shape)} y={tuple(yb.shape)} K={K} '
                      f'mode={args.mode}')
                assert xb.shape[1] == 2, f'输入时间维应为 2，实际 {tuple(xb.shape)}'
                assert yb.shape[1] == K, f'真值时间维应为 K={K}，实际 {tuple(yb.shape)}'

            outs, deltas, xhats = rollout_residual_gan(
                G, R, xb, K, args.alpha, args.mode, device)
            out_flat = outs.reshape(-1, *outs.shape[2:])          # (K*B,1,H,W)
            y_flat = yb.reshape(-1, *yb.shape[2:])
            delta_flat = deltas.reshape(-1, *deltas.shape[2:])
            real_delta = y_flat - xhats.reshape(-1, *xhats.shape[2:])

            # ---------- 判别器更新（判残差场） ----------
            # ⚠️ 假残差必须 detach：否则 loss_d.backward() 会释放 rollout 计算图，
            # 紧接着的生成器反传会报 "backward through the graph a second time"。
            d_real = D(real_delta)
            d_fake = D(delta_flat.detach())
            loss_d = (F.mse_loss(d_real, torch.ones_like(d_real))
                      + F.mse_loss(d_fake, torch.zeros_like(d_fake)))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # ---------- 生成器（残差分支）更新 ----------
            mse = sum(F.mse_loss(outs[k], yb[:, k]) for k in range(K)) / K
            adv = F.mse_loss(D(delta_flat), torch.ones_like(d_fake))   # LSGAN
            rg = rel_grad_loss(out_flat, y_flat)   # 双向锐度，作用于整场输出
            # ⚠️ 量纲：mse≈O(1e-3)，adv/rg≈O(1)，相差约两个数量级。
            # 权重定义为**相对 mse 的比例**，避免对抗项失控。
            loss_g = mse * (1.0 + args.gan_weight * adv + args.grad_weight * rg)
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(R.parameters(), 1.0)
            opt_g.step()
            sched_g.step()

            with torch.no_grad():
                acc['mse'] += float(mse)
                acc['adv'] += float(adv)
                acc['dloss'] += float(loss_d)
                acc['rg'] += float(rg)
                acc['gm_real'] += float(grad_magnitude(y_flat))
                acc['gm_delta'] += float(grad_magnitude(real_delta))
                for k in range(K):
                    per_step[k] += float(F.mse_loss(outs[k], yb[:, k]))
            nb += 1

            if args.log_every and (step + 1) % args.log_every == 0:
                mem = (torch.cuda.max_memory_allocated() / 2 ** 30
                       if torch.cuda.is_available() else 0.0)
                print(f'  ep{epoch} step{step+1}/{steps_per_epoch} '
                      f'mse={acc["mse"]/nb:.5f} adv={acc["adv"]/nb:.4f} '
                      f'D={acc["dloss"]/nb:.4f} rg={acc["rg"]/nb:.4f} '
                      f'mem={mem:.1f}G')

        # ---------------- 验证（单步 MSE，val 集只有 1 帧真值） ----------------
        R.eval()
        vsum, vnb = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                fl = batch_flow(xb)
                prev_K = (xb[:, 1, 0].cpu().numpy()
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
               'train_per_step_mse': (per_step / max(nb, 1)).tolist(),
               'adv': acc['adv'] / max(nb, 1),
               'd_loss': acc['dloss'] / max(nb, 1),
               'rel_grad': acc['rg'] / max(nb, 1),
               'grad_truth': acc['gm_real'] / max(nb, 1),
               'grad_real_delta': acc['gm_delta'] / max(nb, 1),
               'time_s': round(time.time() - t0, 1)}
        hist.append(rec)
        print(f'[EPOCH {epoch}] val_mse={val:.5f} (rmse {rec["val_rmse_K"]:.3f} K) '
              f'adv={rec["adv"]:.4f} D={rec["d_loss"]:.4f} '
              f'逐步MSE={[round(x, 5) for x in rec["train_per_step_mse"]]} '
              f'({rec["time_s"]}s)')

        torch.save({'state_dict': R.state_dict(), 'epoch': epoch,
                    'val_mse': val}, osp.join(ck_dir, 'last.pt'))
        torch.save({'state_dict': D.state_dict(), 'epoch': epoch,
                    'val_mse': val}, osp.join(ck_dir, 'last_D.pt'))
        if val < best:
            best = val
            torch.save({'state_dict': R.state_dict(), 'epoch': epoch,
                        'val_mse': val}, osp.join(ck_dir, 'best.pt'))
            torch.save({'state_dict': D.state_dict(), 'epoch': epoch,
                        'val_mse': val}, osp.join(ck_dir, 'best_D.pt'))
            print(f'  -> 新最优 (val={val:.5f})')

    with open(osp.join(save_dir, 'resgan_history.json'), 'w') as f:
        json.dump(hist, f, indent=2)
    print(f'[DONE] 产物在 {save_dir}')
    print('[HINT] 下一步：python tools/make_residual_preds.py '
          f'--ex-name {args.ex_name} --ckpt {ck_dir}/best.pt')


if __name__ == '__main__':
    main()
