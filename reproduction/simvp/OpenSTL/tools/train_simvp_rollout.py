#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**多步滚动训练 + scheduled sampling**（GAN 的对照方案，不生成假纹理）。

为什么要有这个对照
------------------
GAN 方案实测结果：**所有时效的 RMSE 都未能降低**（LT=18 恶化 34.8%~85.8%），
且长时效锐度比过冲到 2.6~3.0（生成了高频噪声）。根因是 180 分钟后的细节不可预测，
GAN "补细节"本质上是**凭空生成纹理**，而假纹理在 MSE 上一定更差（感知-失真权衡）。

本方案不引入任何生成式损失，只做两件事：
  1. **多步监督**：一次滚动 K 步，对**每一步**都算 MSE（不再只监督第 1 步）
  2. **scheduled sampling**：训练时按概率 p 用**模型自己的预测**（而非真值）作为下一步输入，
     让模型适应推理时的分布，减少曝光偏差（exposure bias）—— 这正是长时效发散的主因

数据来源（关键）
----------------
train 集只有 1 帧真值，**无法**做多步监督。但 **validation 集的 output 有 18 帧**
（`edm_GOES_ch13_validation_dataset.zarr`），可以支撑真正的多步训练。
因此本脚本：
  * 训练/验证：validation 集按索引切分（默认 0~799 训练 / 800~1023 验证）
  * **test 集始终 held-out，不参与任何训练或选点**，与 baseline/GAN 的评估口径一致

⚠️ 注意：这与 baseline/GAN 的训练数据来源不同（train 集 35595 条 vs validation 集 800 条），
因此**不是**严格同数据对照；对照的是"同样热启动、同样梯度步数预算下的改进方向"。
报告时必须注明这一点。

用法::
    python tools/train_simvp_rollout.py --rollout-steps 6 --sched-samp 0.0   # 纯多步
    python tools/train_simvp_rollout.py --rollout-steps 6 --sched-samp 0.5   # + scheduled sampling
"""

import argparse
import json
import math
import os
import os.path as osp
import sys
import time

_HERE = osp.dirname(osp.abspath(__file__))
_OPENSTL_DIR = osp.dirname(_HERE)
if _OPENSTL_DIR not in sys.path:
    sys.path.insert(0, _OPENSTL_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from openstl.datasets.dataloader_satcast import (  # noqa: E402
    SatcastZarrDataset, _SubsetWithAttrs)
from openstl.datasets.utils import create_loader  # noqa: E402
from openstl.models import SimVP_Model  # noqa: E402
from openstl.utils import load_config  # noqa: E402

MODEL_KEYS = ('in_shape', 'hid_S', 'hid_T', 'N_S', 'N_T', 'model_type', 'mlp_ratio',
              'drop', 'drop_path', 'spatio_kernel_enc', 'spatio_kernel_dec',
              'act_inplace')


def rollout_with_ss(model, x, y, steps, p, later_weight=1.0):
    """滚动 K 步，支持 scheduled sampling，返回 (K,B,C,H,W)。

    Args:
        x: (B,2,C,H,W) 条件帧
        y: (B,K,C,H,W) 每一步的真值（K 步都有，来自 validation 集）
        p: 使用**自己预测**作为下一步输入的概率（0=teacher forcing）
    """
    cur, outs = x, []
    b = x.shape[0]
    for k in range(steps):
        # 断言：输入必须恒为 2 帧（模型按 T=2 构建，多一帧会让 MidMetaNet 收到
        # T*hid_S=192 通道而对不上 BN 的 128，报 "running_mean should contain 192"）
        if cur.shape[1] != 2:
            raise AssertionError(
                f'rollout 第 {k} 步输入时间维 = {cur.shape[1]}（应为 2）；'
                f'cur={tuple(cur.shape)} x={tuple(x.shape)} y={tuple(y.shape)}')
        pred_t = model(cur)[:, :1]                 # (B,1,C,H,W)
        outs.append(pred_t.squeeze(1))             # (B,C,H,W)
        if k < steps - 1:
            gt_next = y[:, k].unsqueeze(1)         # 下一时刻的真值帧
            if p > 0:
                # 以概率 p 用模型自己的预测（detach，避免通过采样回传）
                # ⚠️ 必须是 **5 维** (B,1,1,1,1)！
                # 若写成 4 维 (B,1,1,1)，torch.where 的广播会从左补 1 变成 (1,B,1,1,1)，
                # 于是 batch 维 B 被广播到**时间轴**上，nxt 的时间维变成 B，
                # 拼接后 cur 的时间维变成 1+B=3 -> MidMetaNet 收到 T*hid_S=192 通道，
                # 与 BN 的 128 冲突（"running_mean should contain 192 elements not 128"）。
                # （该 bug 在 B=1 时不暴露，B>=2 才炸。）
                use_own = (torch.rand(b, 1, 1, 1, 1, device=x.device) < p)
                nxt = torch.where(use_own, pred_t.detach(), gt_next)
            else:
                nxt = gt_next
            cur = torch.cat([cur[:, 1:], nxt], dim=1)
    return torch.stack(outs, dim=0)


def build_model(cfg_path, in_shape):
    cfg = dict(load_config(cfg_path))
    kw = {k: cfg[k] for k in MODEL_KEYS if k in cfg}
    kw['in_shape'] = list(in_shape)
    return SimVP_Model(**kw), kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=osp.join(
        _OPENSTL_DIR, 'configs', 'satcast', 'SimVP.py'))
    ap.add_argument('--data-root', default='/home/group1/26fall_aiclass/yr/data1/data1/satcast')
    ap.add_argument('--res-dir', default=osp.join(_OPENSTL_DIR, 'work_dirs'))
    ap.add_argument('--ex-name', default='satcast_simvp_rollout')
    ap.add_argument('--train-zarr', default='edm_GOES_ch13_validation_dataset.zarr',
                    help='多步训练数据（该集 output 有 18 帧真值）')
    ap.add_argument('--n-train', type=int, default=800, help='训练用样本数（取前 N 条）')
    ap.add_argument('--n-val', type=int, default=224, help='验证用样本数（取后 M 条）')
    ap.add_argument('--val-source', choices=['validation', 'train'],
                    default='validation',
                    help='验证集来源：validation=从多步训练集中切分（默认）；'
                         'train=改用 train zarr 留出子集，从而训练集可用满全量')
    ap.add_argument('--rollout-steps', type=int, default=6)
    ap.add_argument('--sched-samp', type=float, default=0.0,
                    help='scheduled sampling 概率 p（0=纯 teacher forcing）')
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--init-ckpt', default=None, help='热启动（推荐 baseline 的 best.ckpt）')
    ap.add_argument('--cpu', action='store_true', default=False)
    ap.add_argument('--log-every', type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cpu' if (args.cpu or not torch.cuda.is_available()) else 'cuda')
    in_shape = [2, 1, 256, 256]
    K = args.rollout_steps

    G, kw = build_model(args.config, in_shape)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location='cpu', weights_only=False)
        sd = ck.get('state_dict', ck)
        sd = {k[6:] if k.startswith('model.') else k: v for k, v in sd.items()}
        missing = set(G.state_dict()) - set(sd)
        if missing:
            raise RuntimeError(f'热启动权重不匹配：missing={sorted(missing)[:5]}')
        G.load_state_dict(sd, strict=True)
        print(f'[INIT] 已从 {args.init_ckpt} 热启动')
    G.to(device)

    # 数据源：多步监督要求 output 有 ≥K 帧真值，只有 validation 集满足（18 帧）。
    # 两种配置：
    #   --val-source validation（默认）：validation 集自身切分训练/验证
    #   --val-source train：训练集用满**全部** validation 样本，
    #     验证集改用 train zarr 的末尾留出子集（仅 1 帧真值 → 按单步 MSE 验证）。
    #     这样能解除"训练集最多 1024 条"的上限，用于检验收益是否只是小样本假象。
    zarr_path = osp.join(args.data_root, args.train_zarr)
    full = SatcastZarrDataset(zarr_path, pre_seq_length=2, aft_seq_length=K)
    n_total = len(full)
    if args.val_source == 'train':
        n_tr = min(args.n_train, n_total)
        train_set = _SubsetWithAttrs(full, range(0, n_tr))
        train_zarr = SatcastZarrDataset(
            osp.join(args.data_root, 'edm_GOES_ch13_train_dataset.zarr'),
            pre_seq_length=2, aft_seq_length=1)
        n_va = min(args.n_val, len(train_zarr))
        val_set = _SubsetWithAttrs(
            train_zarr, range(len(train_zarr) - n_va, len(train_zarr)))
        print(f'[DATA] 训练 = validation 集前 {n_tr} 条（output {K} 帧真值，全量）；'
              f'验证 = train 集末尾 {n_va} 条（单步 MSE，1 帧真值）；'
              f'test 集保持 held-out')
    else:
        n_tr = min(args.n_train, n_total - args.n_val)
        n_va = min(args.n_val, n_total - n_tr)
        train_set = _SubsetWithAttrs(full, range(0, n_tr))
        val_set = _SubsetWithAttrs(full, range(n_total - n_va, n_total))
        print(f'[DATA] {osp.basename(zarr_path)}  总 {n_total} 条 -> '
              f'train {len(train_set)} / val {len(val_set)}（test 集保持 held-out）')

    persistent = args.num_workers > 0
    train_loader = create_loader(
        train_set, batch_size=args.batch_size, shuffle=True, is_training=True,
        pin_memory=True, drop_last=True, num_workers=args.num_workers,
        persistent_workers=persistent)
    val_loader = create_loader(
        val_set, batch_size=4, shuffle=False, is_training=False,
        pin_memory=True, drop_last=False, num_workers=args.num_workers,
        persistent_workers=persistent)

    opt = torch.optim.AdamW(G.parameters(), lr=args.lr, weight_decay=0)
    steps_per_epoch = len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, steps_per_epoch * args.epochs),
        pct_start=0.1)

    save_dir = osp.join(args.res_dir, args.ex_name)
    ck_dir = osp.join(save_dir, 'checkpoints')
    os.makedirs(ck_dir, exist_ok=True)
    with open(osp.join(save_dir, 'rollout_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print('=' * 74)
    print('SimVP 多步滚动训练 + scheduled sampling（GAN 对照方案）')
    print('=' * 74)
    print(f'[MODEL] {kw}  参数 {sum(p.numel() for p in G.parameters())/1e6:.3f} M')
    print(f'[TRAIN] rollout_steps={K}  sched_samp={args.sched_samp}  '
          f'batch={args.batch_size}  epochs={args.epochs}  steps/epoch={steps_per_epoch}')
    print(f'[LOSS ] 多步 MSE：对全部 {K} 步逐一求 MSE 后取平均（无生成式损失）')
    print(f'[SAVE ] {save_dir}')
    print('=' * 74)

    best_val, history = float('inf'), []
    for epoch in range(args.epochs):
        G.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        # 逐时效损失累加（用于看哪一步最难）
        per_step = np.zeros(K)
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if epoch == 0 and step == 0:
                print(f'[DEBUG] batch: x={tuple(xb.shape)} y={tuple(yb.shape)}  '
                      f'K={K} p={args.sched_samp}')
                assert xb.shape[1] == 2, (
                    f'输入时间维应为 2（模型按 T=2 构建），实际 {tuple(xb.shape)}；'
                    f'若不是 2 说明数据/切分有误')
                assert yb.shape[1] == K, (
                    f'真值时间维应为 K={K}，实际 {tuple(yb.shape)}')
            preds = rollout_with_ss(G, xb, yb, K, args.sched_samp)
            loss = sum(F.mse_loss(preds[k], yb[:, k]) for k in range(K)) / K
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for k in range(K):
                    per_step[k] += float(F.mse_loss(preds[k], yb[:, k]))
            tot += float(loss)
            nb += 1
            if args.log_every and (step + 1) % args.log_every == 0:
                print(f'  ep{epoch} step{step+1} loss={tot/nb:.5f} '
                      f'mem={torch.cuda.max_memory_allocated()/2**30:.1f}G')

        G.eval()
        vsum, vnb = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                # kv 动态：validation 集为 K 帧；train 集仅 1 帧
                kv = yb.shape[1]
                p = rollout_with_ss(G, xb, yb, kv, 0.0)      # 验证时不用 SS
                vsum += float(sum(F.mse_loss(p[k], yb[:, k])
                                  for k in range(kv)) / kv)
                vnb += 1
        val = vsum / max(vnb, 1)
        rec = {'epoch': epoch, 'train_loss': tot / max(nb, 1),
               'val_loss_multistep': val,
               'val_rmse_K': math.sqrt(val) * 19.32967519050003,
               'per_step_train_mse': (per_step / max(nb, 1)).tolist(),
               'time_s': round(time.time() - t0, 1)}
        history.append(rec)
        print(f'[EPOCH {epoch}] val_multistep_mse={val:.5f} '
              f'(rmse {rec["val_rmse_K"]:.3f}K)  '
              f'逐步MSE={[round(x,5) for x in rec["per_step_train_mse"]]}  '
              f'({rec["time_s"]}s)')

        torch.save({'state_dict': G.state_dict(), 'epoch': epoch, 'val_mse': val},
                   osp.join(ck_dir, 'last.pt'))
        if val < best_val:
            best_val = val
            torch.save({'state_dict': G.state_dict(), 'epoch': epoch,
                        'val_mse': val}, osp.join(ck_dir, 'best.pt'))
            torch.save({'state_dict': {f'model.{k}': v for k, v in G.state_dict().items()},
                        'epoch': epoch, 'val_mse': val}, osp.join(ck_dir, 'best.ckpt'))
            print(f'  -> 新最优 (val={val:.5f})，已同步保存 best.ckpt')
        with open(osp.join(save_dir, 'rollout_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print('[DONE] 产物在 ' + save_dir)


if __name__ == '__main__':
    main()
