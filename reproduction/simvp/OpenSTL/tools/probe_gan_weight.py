#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**GAN 权重筛选探针**：在同一架构、同数据、同步数下，测不同 `gan_weight`
对「残差拟合 / 锐度 / 覆盖率 / 最终产品精度」的影响，用于选定权重。

为什么需要这个脚本
-------------------
此前的权重探针只记录了残差 MSE，而 MSE 在该区间几乎不随权重变化
（step100 均约 0.0078），**无法区分权重好坏**。真正有区分度的是：
  * 锐度：`grad(B输出) / grad(真实残差)` —— GAN 是否让残差产品变噪
  * 覆盖率：`(â+B)` 的冷芯占比 / 真值冷芯占比 —— GAN 的真实价值所在
  * 最终产品 RMSE：`â_1 + B_1` vs `真值_1`

方法
----
对每个候选权重：
  1. 用**固定** train 子集训练 B 固定步数（同 seed，保证唯一变量是权重）
  2. 在**留出的** train 子集上评估上述四类指标
  3. 剪枝前打印，全部跑完汇总成表

⚠️ 这是**筛选工具**，不是结论。最终权重必须经完整训练 + test 集 8 指标确认。

用法::

    python tools/probe_gan_weight.py --weights 0.25,0.1,0.05,0.02,0.01,0 \
        --train-samples 4096 --eval-samples 512 --steps 256
"""

import argparse
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

from train_residual_product import ResidualProduct, load_frozen_A  # noqa: E402
from train_simvp_gan import PatchDiscriminator, build_model, grad_magnitude, rollout  # noqa: E402

COLD = 235.0


@torch.no_grad()
def evaluate(A, B, loader, device, n_lead):
    """在留出子集上算四类指标（只用第 1 步，因 train 只有 1 帧真值）。"""
    acc = {'mse': 0.0, 'mse_zero': 0.0, 'n': 0,
           'g_b': 0.0, 'g_r': 0.0,
           'cov_p': 0.0, 'cov_t': 0.0, 'nf': 0,
           'final_mse': 0.0, 'base_mse': 0.0}
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        ahat = A(xb)[:, :1].squeeze(1)                 # (B,1,H,W)
        real_resid = yb[:, 0] - ahat
        xin = torch.cat([xb[:, 0], xb[:, 1], ahat], dim=1)
        b_out = B(xin, 1)
        final = ahat + b_out

        acc['mse'] += float(F.mse_loss(b_out, real_resid)) * xb.shape[0]
        acc['mse_zero'] += float(real_resid.pow(2).mean()) * xb.shape[0]
        acc['n'] += xb.shape[0]
        acc['g_b'] += float(grad_magnitude(b_out))
        acc['g_r'] += float(grad_magnitude(real_resid))
        # 覆盖率（亮温 K 空间）
        fk = final[:, 0] * STD_K + MEAN_K
        tk = yb[:, 0, 0] * STD_K + MEAN_K
        acc['cov_p'] += float((fk <= COLD).float().mean())
        acc['cov_t'] += float((tk <= COLD).float().mean())
        acc['nf'] += 1
        acc['final_mse'] += float(F.mse_loss(final, yb[:, 0])) * xb.shape[0]
        acc['base_mse'] += float(F.mse_loss(ahat, yb[:, 0])) * xb.shape[0]
    n = max(acc['n'], 1)
    return {
        'resid_mse': acc['mse'] / n,
        'resid_mse_zero': acc['mse_zero'] / n,
        'resid_rmse_K': float(np.sqrt(acc['mse'] / n) * STD_K),
        'beats_zero': acc['mse'] < acc['mse_zero'],
        'sharp_ratio': acc['g_b'] / max(acc['g_r'], 1e-9),
        'cov_ratio': (acc['cov_p'] / max(acc['nf'], 1)) / max(acc['cov_t'] / max(acc['nf'], 1), 1e-9),
        'final_rmse_K': float(np.sqrt(acc['final_mse'] / n) * STD_K),
        'base_rmse_K': float(np.sqrt(acc['base_mse'] / n) * STD_K),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=osp.join(
        _OPENSTL_DIR, 'configs', 'satcast', 'SimVP.py'))
    ap.add_argument('--data-root',
                    default=os.environ.get(
                        'SATCAST_ROOT',
                        '/home/group1/26fall_aiclass/yr/data1/data1/satcast'))
    ap.add_argument('--a-ckpt', default=osp.join(
        _OPENSTL_DIR, 'work_dirs', 'satcast_simvp_full', 'checkpoints', 'best.ckpt'))
    ap.add_argument('--weights', default='0.25,0.1,0.05,0.02,0.01,0')
    ap.add_argument('--train-samples', type=int, default=4096)
    ap.add_argument('--eval-samples', type=int, default=512)
    ap.add_argument('--steps', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--rollout-steps', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true', default=False)
    args = ap.parse_args()

    device = torch.device('cpu' if (args.cpu or not torch.cuda.is_available())
                          else 'cuda')
    K = args.rollout_steps
    weights = [float(w) for w in args.weights.split(',') if w.strip()]

    full = SatcastZarrDataset(
        osp.join(args.data_root, 'edm_GOES_ch13_train_dataset.zarr'),
        pre_seq_length=2, aft_seq_length=1)
    n_total = len(full)
    n_ev = args.eval_samples
    n_tr = min(args.train_samples, n_total - n_ev)
    # 训练用前 n_tr 条，评估用**末尾** n_ev 条（不重叠）
    tr_set = _SubsetWithAttrs(full, range(0, n_tr))
    ev_set = _SubsetWithAttrs(full, range(n_total - n_ev, n_total))
    tr_loader = create_loader(tr_set, batch_size=args.batch_size, shuffle=True,
                              is_training=True, pin_memory=True, drop_last=True,
                              num_workers=8, persistent_workers=True)
    ev_loader = create_loader(ev_set, batch_size=16, shuffle=False,
                              is_training=False, pin_memory=True, drop_last=False,
                              num_workers=4)
    print(f'[DATA] 训练 {len(tr_set)} 条（前）｜评估 {len(ev_set)} 条（末尾，不重叠）')

    A = load_frozen_A(args.config, args.a_ckpt, [2, 1, 256, 256], device)

    results = []
    for w in weights:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        B = ResidualProduct(in_ch=3, base=48, n_lead=K).to(device)
        D = PatchDiscriminator(in_ch=1, base=64, n_layers=4).to(device)
        opt_b = torch.optim.AdamW(B.parameters(), lr=args.lr, weight_decay=0)
        opt_d = torch.optim.AdamW(D.parameters(), lr=1e-4, weight_decay=0)
        B.train()
        D.train()
        t0 = time.time()
        step = 0
        done = False
        while not done:
            for xb, yb in tr_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.no_grad():
                    ahat = rollout(A, xb, K)
                f_in, f_t = xb[:, 0], xb[:, 1]
                real_resid = yb[:, 0] - ahat[0]
                b_outs = torch.stack(
                    [B(torch.cat([f_in, f_t, ahat[k]], dim=1), k + 1)
                     for k in range(K)], dim=0)
                b_flat = b_outs.reshape(-1, *b_outs.shape[2:])
                d_real = D(real_resid)
                d_fake = D(b_flat.detach())
                loss_d = (F.mse_loss(d_real, torch.ones_like(d_real))
                          + F.mse_loss(d_fake, torch.zeros_like(d_fake)))
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
                mse = F.mse_loss(b_outs[0], real_resid)
                adv = F.mse_loss(D(b_flat), torch.ones_like(d_fake))
                loss_b = mse * (1.0 + w * adv)
                opt_b.zero_grad(set_to_none=True)
                loss_b.backward()
                torch.nn.utils.clip_grad_norm_(B.parameters(), 1.0)
                opt_b.step()
                step += 1
                if step >= args.steps:
                    done = True
                    break

        B.eval()
        m = evaluate(A, B, ev_loader, device, K)
        m['weight'] = w
        m['time_s'] = round(time.time() - t0, 1)
        results.append(m)
        print(f'[w={w:<5}] 残差RMSE={m["resid_rmse_K"]:.3f}K '
              f'(B=0 {np.sqrt(m["resid_mse_zero"])*STD_K:.3f}K, '
              f'{"优于" if m["beats_zero"] else "劣于"}) '
              f'锐度比={m["sharp_ratio"]:.3f} 覆盖率比={m["cov_ratio"]:.3f} '
              f'最终RMSE={m["final_rmse_K"]:.3f}K '
              f'(A单独 {m["base_rmse_K"]:.3f}K)  [{m["time_s"]}s]', flush=True)

    # ---------------- 汇总表 ----------------
    print('\n' + '=' * 96)
    print('GAN 权重筛选汇总（train-only，留出子集评估，唯一变量=权重）')
    print('=' * 96)
    hdr = (f'{"weight":>7}{"残差RMSE":>10}{"vs B=0":>9}{"锐度比":>8}'
           f'{"覆盖率比":>9}{"最终RMSE":>10}{"vs A单独":>9}')
    print(hdr)
    print('-' * 96)
    base_rmse = None
    for r in results:
        if r['weight'] == 0:
            base_rmse = r['final_rmse_K']
    for r in results:
        vs = ('基准' if r['weight'] == 0
              else f'{(r["final_rmse_K"]/base_rmse-1)*100:+.2f}%')
        bz = '优于' if r['beats_zero'] else '劣于'
        print(f'{r["weight"]:>7}{r["resid_rmse_K"]:>10.3f}{bz:>9}'
              f'{r["sharp_ratio"]:>8.3f}{r["cov_ratio"]:>9.3f}'
              f'{r["final_rmse_K"]:>10.3f}{vs:>9}')
    print('-' * 96)
    print('锐度比 = grad(B输出)/grad(真实残差)，越接近 1 越像真实残差；'
          '>1 = 比真实残差更噪')
    print('覆盖率比 = (â+B)冷芯占比 / 真值冷芯占比，越接近 1 越好')
    print('最终RMSE = â_1 + B_1 vs 真值_1（亮温 K），vs A单独 是相对无 GAN 的变化')


if __name__ == '__main__':
    main()
