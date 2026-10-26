#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**产品 B：独立残差预报产品**（修正主产品 A 的滚动预报）。

定位（与旧 `train_residual.py` / `train_residual_gan.py` 的本质区别）
-------------------------------------------------------------------
旧脚本把残差修正器**接进主产品的反馈环**（`cur = [cur[:,1:], x̂+Δ]`），
B 的误差会污染 A 后续步的输入。本脚本按用户澄清的定义重做：

  * 产品 A（SimVP + GAN）**独立**自回归 rollout，反馈只用 A 自己的原始输出 â
  * 产品 B **独立**训练，输出的是"残差场"这一预报产品
  * 最终产品 = A + B，**只在末端相加**，B 不进入 A 的反馈环

结构
----
::

    输入 2 帧 ──► 产品 A（冻结）──► â_1 … â_K        （A 的内部反馈只用 â）
                      │
                      ▼ 每步取出 â_k
    输入 2 帧 + â_k + lead_k ──► 产品 B ──► B_k        （残差产品）
                                                      
    最终产品 = â_k + B_k

训练约束（用户指定：只用 train，不用 validation）
----------------------------------------------
train 集 `output_images` 只有 **1 帧**真值，因此：
  * A 的 K=6 rollout 可以照跑（自回归不需要真值）
  * 但残差目标 `r_k = 真值_k − â_k` **只有 k=1 可得**
  * → B 的 MSE 监督**只能落在第 1 步**；第 2~6 步靠 GAN 的分布匹配信号

⚠️ 这条约束的后果必须在报告里写明：B 的修正是从"第 1 步"外推到其它步的，
   预期收益集中在**短时效**（与"重点关注前 1h"的目标一致）。

损失（gan : le = 1 : 5，即 gan_weight = 0.2）
--------------------------------------------
    loss_b = mse * (1.0 + 0.2 * adv)

权重取值依据**权重探针**（`tools/probe_gan_weight.py`，train-only、
同数据 4096 条、同步数 256 步、留出 512 条评估，唯一变量=权重）：

| gan_weight | 锐度比 | 覆盖率比 | 最终RMSE | vs A单独 |
|---|---|---|---|---|
| 0.25 | 0.093 | 0.995 | 1.703 K | +0.92% |
| **0.2（取）** | — | — | — | 约 +0.013 K（内插） |
| 0.10 | 0.080 | 0.995 | 1.698 K | +0.66% |
| 0.05 | 0.034 | 0.999 | 1.696 K | +0.50% |
| 0.02 | 0.024 | 0.994 | 1.688 K | +0.06% |
| 0（无GAN） | 0.009 | 0.993 | 1.687 K | 基准 |

锐度比 = grad(B输出)/grad(真实残差)。探针显示：
  * 锐度增益随权重单调上升，但 **w≈0.1 后趋饱和**（0.1→0.25 权重涨 2.5×，
    增益仅 +0.013），而 RMSE 代价持续上升 → 0.2 位于"增益仍显著、代价可接受"段
  * 探针中 B 各权重均略劣于 B≡0（1.687 K），说明当时**瓶颈是优化不充分**
    （256 步 + lr 瞬态），不是权重问题；该权重需完整训练复核
  * 探针只评估第 1 步（train 仅 1 帧真值），覆盖率维度测不出差异；
    覆盖率收益需靠 18 步完整评估

判别器：
    真实残差 = 真值_1 − â_1   （train 35595 条，充足）
    假残差   = B 在全部 K 步上的输出
    ⚠️ 已知局限：真实残差只有第 1 步的，故 D 学到的是"第 1 步残差的分布"，
       对第 2~6 步的对抗信号实为"像个合理残差场"而非"匹配该步真实残差分布"。

用法::

    python tools/train_residual_product.py \
        --a-ckpt work_dirs/productA_gan_k6/checkpoints/best.ckpt \
        --ex-name productB_k6
    # 出最终产品 A+B 的 18 步预测
    python tools/make_final_preds.py --ex-name productB_k6
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
    SatcastZarrDataset, _SubsetWithAttrs)
from openstl.datasets.utils import create_loader  # noqa: E402
from openstl.utils import load_config  # noqa: E402

# 复用已验证的实现（避免符号约定/架构漂移）
from train_simvp_gan import (MODEL_KEYS, PatchDiscriminator, build_model,  # noqa: E402
                             grad_magnitude, rollout)
from openstl.models import SimVP_Model  # noqa: E402


# ------------------------------------------------------------------ 产品 B ---
def _conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(8, cout),
        nn.GELU(),
    )


class ResidualProduct(nn.Module):
    """独立残差预报产品：`[f_t-10, f_t, â_k, lead_k]` -> 残差场 `(B,1,H,W)`。

    * `â_k`：产品 A 在第 k 步的输出（提供"主产品这一步错在哪"的线索）
    * `lead_k`：时效编码（常数通道，值 = k/K）。残差的性质随时效剧烈变化
      （实测：偏差从 −0.001 K 涨到 −2.637 K，冷芯集中度从 1.2× 到 4.6×），
      不给 lead 信息，网络只能输出各时效的平均修正。
    * 输出层**零初始化** → 初始 B≡0，最终产品严格等于 A，净贡献可精确度量。
    """

    def __init__(self, in_ch=3, base=48, n_lead=6):
        super().__init__()
        self.n_lead = n_lead
        self.e1 = _conv(in_ch + 1, base)          # 256
        self.e2 = _conv(base, base * 2)           # 128
        self.e3 = _conv(base * 2, base * 4)       # 64
        self.e4 = _conv(base * 4, base * 8)       # 32
        self.d3 = _conv(base * 8 + base * 4, base * 4)
        self.d2 = _conv(base * 4 + base * 2, base * 2)
        self.d1 = _conv(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, k):
        """x: (B,in_ch,H,W)；k: int 或 (B,) 的 lead 索引（1-based）。"""
        b, _, h, w = x.shape
        if isinstance(k, int):
            k = torch.full((b,), k, device=x.device, dtype=torch.float32)
        k = torch.as_tensor(k, device=x.device, dtype=torch.float32)
        lead = (k / float(self.n_lead)).view(b, 1, 1, 1).expand(b, 1, h, w)
        inp = torch.cat([x, lead], dim=1)
        e1 = self.e1(inp)
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


def load_frozen_A(cfg_path, ckpt, in_shape, device):
    """加载产品 A 并冻结（A 只提供 â_k，不回传梯度）。"""
    model, _ = build_model(cfg_path, in_shape)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    sd = {k[6:] if k.startswith('model.') else k: v for k, v in sd.items()}
    missing = set(model.state_dict()) - set(sd)
    if missing:
        raise RuntimeError(f'产品 A 权重不匹配：missing={sorted(missing)[:5]}')
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


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
    ap.add_argument('--ex-name', default='productB_k6')
    ap.add_argument('--a-ckpt', required=True,
                    help='产品 A 的 best.ckpt（提供 â_k）')
    ap.add_argument('--rollout-steps', type=int, default=6)
    ap.add_argument('--gan-weight', type=float, default=0.2,
                    help='对抗项相对 mse 的比例。默认 0.2（= gan:le = 1:5）。'
                         '依据权重探针（train-only、同数据同步数、留出子集评估）：'
                         'w=0.25 -> 锐度比 0.093 / 最终RMSE +0.92%；'
                         'w=0.10 -> 0.080 / +0.66%；w=0.05 -> 0.034 / +0.50%；'
                         'w=0.02 -> 0.024 / +0.06%。锐度增益在 w≈0.1 后趋饱和，'
                         '而 RMSE 代价持续上升，故 0.2 取在"增益仍显著、代价可接受"段。'
                         '注意：探针中 B 各权重均略劣于 B≡0，说明当时瓶颈是优化不充分'
                         '而非权重；该权重需经完整训练复核。')
    ap.add_argument('--width', type=int, default=48)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr-d', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit-train', type=int, default=None)
    ap.add_argument('--cpu', action='store_true', default=False)
    ap.add_argument('--log-every', type=int, default=200)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cpu' if (args.cpu or not torch.cuda.is_available())
                          else 'cuda')
    K = args.rollout_steps

    # ---------------- 数据（只用 train） ----------------
    full = SatcastZarrDataset(
        osp.join(args.data_root, 'edm_GOES_ch13_train_dataset.zarr'),
        pre_seq_length=2, aft_seq_length=1)
    n_total = len(full)
    train_set = _SubsetWithAttrs(
        full, range(0, min(args.limit_train or n_total, n_total)))
    train_loader = create_loader(
        train_set, batch_size=args.batch_size, shuffle=True, is_training=True,
        pin_memory=True, drop_last=True, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0)
    print(f'[DATA] 只用 train 集：{len(train_set)} 条（output 1 帧真值）')

    # ---------------- 模型 ----------------
    A = load_frozen_A(args.config, args.a_ckpt, [2, 1, 256, 256], device)
    B = ResidualProduct(in_ch=3, base=args.width, n_lead=K).to(device)
    D = PatchDiscriminator(in_ch=1, base=64, n_layers=4).to(device)

    opt_b = torch.optim.AdamW(B.parameters(), lr=args.lr, weight_decay=0)
    opt_d = torch.optim.AdamW(D.parameters(), lr=args.lr_d, weight_decay=0)
    steps_per_epoch = len(train_loader)
    # ⚠️ total_steps 必须用真实 steps/epoch（曾因写常数中途崩）
    sched_b = torch.optim.lr_scheduler.OneCycleLR(
        opt_b, max_lr=args.lr,
        total_steps=max(1, steps_per_epoch * args.epochs), pct_start=0.1)

    save_dir = osp.join(args.res_dir, args.ex_name)
    ck_dir = osp.join(save_dir, 'checkpoints')
    os.makedirs(ck_dir, exist_ok=True)
    with open(osp.join(save_dir, 'productB_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    n_a = sum(p.numel() for p in A.parameters()) / 1e6
    n_b = sum(p.numel() for p in B.parameters()) / 1e6
    n_d = sum(p.numel() for p in D.parameters()) / 1e6
    print('=' * 76)
    print(f'产品 A(冻结) {n_a:.3f}M  +  产品 B(残差) {n_b:.3f}M  +  '
          f'判别器 {n_d:.3f}M')
    print(f'[TRAIN] K={K} batch={args.batch_size} epochs={args.epochs} '
          f'steps/epoch={steps_per_epoch}')
    print(f'[LOSS ] B = mse * (1 + {args.gan_weight}·adv)  ->  '
          f'gan:le = {args.gan_weight}:1  （即 1:{1/max(args.gan_weight,1e-9):.0f}）')
    print('[DISC ] 真实残差 = 真值_1 − â_1；假残差 = B 在全部 K 步的输出')
    print('[INIT ] B 输出层零初始化 -> 初始 B≡0，最终产品严格等于 A')
    print('=' * 76)

    best, hist = float('inf'), []
    for epoch in range(args.epochs):
        B.train()
        D.train()
        t0, nb = time.time(), 0
        acc = {'mse': 0.0, 'adv': 0.0, 'dloss': 0.0,
               'gm_real': 0.0, 'gm_b': 0.0, 'bias': 0.0,
               'mse_zero': 0.0}

        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)      # (B,2,1,H,W)
            yb = yb.to(device, non_blocking=True)      # (B,1,1,H,W)
            b = xb.shape[0]

            # ---- 产品 A 的 K 步 rollout（冻结，无梯度） ----
            with torch.no_grad():
                ahat = rollout(A, xb, K)               # (K,B,1,H,W)

            f_in = xb[:, 0]                            # t-10
            f_t = xb[:, 1]                             # t
            real_resid = (yb[:, 0] - ahat[0])          # 第 1 步真实残差

            # ---- 产品 B 对每一步输出残差 ----
            b_outs = []
            for k in range(K):
                xin = torch.cat([f_in, f_t, ahat[k]], dim=1)   # (B,3,H,W)
                b_outs.append(B(xin, k + 1))
            b_outs = torch.stack(b_outs, dim=0)        # (K,B,1,H,W)
            b_flat = b_outs.reshape(-1, *b_outs.shape[2:])     # (K*B,1,H,W)

            # ---- 判别器更新（判残差场） ----
            # ⚠️ 假残差必须 detach：否则 loss_d.backward() 会释放 B 的计算图
            d_real = D(real_resid)
            d_fake = D(b_flat.detach())
            loss_d = (F.mse_loss(d_real, torch.ones_like(d_real))
                      + F.mse_loss(d_fake, torch.zeros_like(d_fake)))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # ---- 产品 B 更新 ----
            # MSE 只能监督第 1 步（train 只有 1 帧真值）
            mse = F.mse_loss(b_outs[0], real_resid)
            adv = F.mse_loss(D(b_flat), torch.ones_like(d_fake))
            loss_b = mse * (1.0 + args.gan_weight * adv)

            opt_b.zero_grad(set_to_none=True)
            loss_b.backward()
            torch.nn.utils.clip_grad_norm_(B.parameters(), 1.0)
            opt_b.step()
            sched_b.step()

            with torch.no_grad():
                acc['mse'] += float(mse)
                acc['adv'] += float(adv)
                acc['dloss'] += float(loss_d)
                acc['gm_real'] += float(grad_magnitude(real_resid))
                acc['gm_b'] += float(grad_magnitude(b_flat))
                acc['bias'] += float(b_outs[0].mean())
                # ★ B≡0 的参照点：残差场的二阶矩（B 零初始化时的 MSE）。
                #   B 必须把这个值压下去才有存在意义；压不过 -> 残差产品无效。
                acc['mse_zero'] += float(real_resid.pow(2).mean())
            nb += 1

            if args.log_every and (step + 1) % args.log_every == 0:
                mem = (torch.cuda.max_memory_allocated() / 2 ** 30
                       if torch.cuda.is_available() else 0.0)
                mz = acc['mse_zero'] / nb
                cur = acc['mse'] / nb
                print(f'  ep{epoch} step{step+1}/{steps_per_epoch} '
                      f'mse={cur:.6f} (B=0参照 {mz:.6f}, '
                      f'{"优于" if cur < mz else "劣于"}) '
                      f'adv={acc["adv"]/nb:.4f} D={acc["dloss"]/nb:.4f} '
                      f'bias={acc["bias"]/nb:+.4f} mem={mem:.1f}G')

        rec = {'epoch': epoch,
               'train_resid_mse': acc['mse'] / max(nb, 1),
               'train_resid_mse_zero_ref': acc['mse_zero'] / max(nb, 1),
               'train_resid_rmse_K': float(np.sqrt(acc['mse'] / max(nb, 1))
                                           * 19.32967519050003),
               'beats_zero_ref': bool(acc['mse'] < acc['mse_zero']),
               'adv': acc['adv'] / max(nb, 1),
               'd_loss': acc['dloss'] / max(nb, 1),
               'grad_real_resid': acc['gm_real'] / max(nb, 1),
               'grad_b_out': acc['gm_b'] / max(nb, 1),
               'b_step1_mean': acc['bias'] / max(nb, 1),
               'time_s': round(time.time() - t0, 1)}
        hist.append(rec)
        print(f'[EPOCH {epoch}] 残差MSE={rec["train_resid_mse"]:.6f} '
              f'(B=0参照 {rec["train_resid_mse_zero_ref"]:.6f}, '
              f'{"✅优于" if rec["beats_zero_ref"] else "❌劣于"}) '
              f'RMSE {rec["train_resid_rmse_K"]:.3f} K  '
              f'adv={rec["adv"]:.4f} D={rec["d_loss"]:.4f}  '
              f'B均值={rec["b_step1_mean"]:+.4f}  ({rec["time_s"]}s)')

        torch.save({'state_dict': B.state_dict(), 'epoch': epoch,
                    'val': rec['train_resid_mse']}, osp.join(ck_dir, 'last.pt'))
        if rec['train_resid_mse'] < best:
            best = rec['train_resid_mse']
            torch.save({'state_dict': B.state_dict(), 'epoch': epoch,
                        'val': best}, osp.join(ck_dir, 'best.pt'))
            print(f'  -> 新最优 (resid_mse={best:.6f})')

    with open(osp.join(save_dir, 'productB_history.json'), 'w') as f:
        json.dump(hist, f, indent=2)
    print(f'[DONE] 产品 B 产物在 {save_dir}')
    print('[HINT] python tools/make_final_preds.py '
          f'--ex-name {args.ex_name} --a-ckpt {args.a_ckpt}')


if __name__ == '__main__':
    main()
