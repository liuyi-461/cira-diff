#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**GAN + scheduled sampling 联合训练**（对抗损失与曝光偏差修正同时启用）。

为什么需要这个脚本
------------------
此前跑过两条**独立**的改良路线，但没有组合：

| 路线 | 脚本 | 数据源 | rollout 方式 | 损失 |
|---|---|---|---|---|
| GAN | `train_simvp_gan.py` | train zarr（35595 条，**仅 1 帧真值**） | 全自回归（恒喂自身预测，等价 p=1） | `mse*(1+gan_w*adv+grad_w*rel_grad)`，mse 仅第 1 步 |
| 多步+SS | `train_simvp_rollout.py` | validation zarr（1024 条，**18 帧真值**） | `rollout_with_ss`（按概率 p 混真值） | 全部 K 步 MSE 取平均 |

用户要求"GAN+SS 同时启用"。**关键决策：以 `train_simvp_rollout.py` 为基座**，
因为 scheduled sampling 与多步 MSE 都**必须有多帧真值**，只有 validation zarr 满足；
在 train zarr 上 SS 在数学上无法成立（没有 t+20min 之后的真值可用作 teacher signal）。

合并后的损失（K 步真值全部可用，比原 GAN 更强）
--------------------------------------------
    loss_g = multistep_mse * (1.0 + gan_weight * adv + grad_weight * rel_grad)

    multistep_mse = mean_k MSE(preds[k], y[k])     # K 步真值全部可用（原 GAN 只能用第 1 步）
    adv           = LSGAN(D(preds_flat), 1)         # preds 来自 SS rollout = 「滚动预报结果」
    rel_grad      = ((grad_mag(preds) - grad_mag(y)) / (grad_mag(y)+eps))**2

判别器的真实样本直接用 `y`（K 帧真值），与假样本（K 步滚动预报）**数量对称、口径一致**，
比原 GAN 的「2 帧输入 + 1 帧真值」更充分。

判别器设计（沿用已验证版本）
--------------------------
`PatchDiscriminator`：4 层 Conv(stride=2) + InstanceNorm + LeakyReLU，输出 **patch 级** logits
(B,1,H/16,W/16)。每个输出位置对应输入的一个感受野块，因此能对**局部纹理是否模糊**给出
细粒度监督，比单一标量更利于抗模糊。已加 **谱归一化**约束 Lipschitz 常数（未加时实测
判别器会把生成器推向高频噪声，锐度比飙到 2.7）。

用户思路的对应关系
----------------
1. 训练时**滚动 K 步**，把**滚动结果**（而非单步结果）送入判别器 —— `rollout_with_ss` 的
   输出 `preds` 即滚动结果，第 2 步起都依赖模型自己的预测；
2. 判别器判断一帧是否**真实**：真实（清晰）→ loss 低，预报（模糊）→ loss 高；
3. 对抗损失反传到生成器，约束它产生**越来越真实（不模糊）**的滚动结果。

⚠️ 必须原样保留的避坑逻辑（漏掉任一条都会复现已修复的 bug）
----------------------------------------------------------
* `torch.where` 的条件张量必须是 **5 维** `(B,1,1,1,1)`。写成 4 维会因广播"从左补 1"变成
  `(1,B,1,1,1)`，把 batch 维广播到**时间轴**，`cur` 时间维变成 3 →
  MidMetaNet 收到 `T*hid_S=192` 通道与 BN 的 128 冲突。（B=1 不暴露，B≥2 才炸）
* rollout 每步入口断言 `cur.shape[1] == 2`。
* 收集预测时 `model(cur)[:, :1]` 后必须 `squeeze(1)`，否则带时间维：
  验证时与真值广播使指标虚高，且 `grad_magnitude` 在 5 维上取到空张量 → NaN。
* `OneCycleLR(total_steps = len(train_loader) * epochs)`，不可拍脑袋写常数。
* 权重语义是「**相对 mse 的比例**」：mse≈6.6e-3 而 adv/rg 是 O(1)，相差约 75 倍；
  若写成 `mse + w*adv`，即使 w=0.01 也会与 mse 同量级并带偏模型。
* 生成器梯度裁剪 `clip_grad_norm_(1.0)`。
* 另存带 `model.` 前缀的 `best.ckpt`，供 `tools/eval_simvp_baseline.py` 直接加载。
* `sys.path` 自举；`--data-root` 优先取环境变量 `SATCAST_ROOT`。

用法::
    # 与既有 SS 全量臂完全同配置，仅叠加对抗损失（构成干净的 2×2 消融）
    python tools/train_simvp_gan_ss.py \
        --ex-name satcast_simvp_gan_ss \
        --rollout-steps 6 --sched-samp 0.5 \
        --gan-weight 0.05 --grad-weight 0.05 \
        --n-train 1024 --n-val 512 --val-source train \
        --epochs 16 --batch-size 2 \
        --init-ckpt work_dirs/satcast_simvp_full/checkpoints/best.ckpt

⚠️ 显存：滚动 K 步 = K 倍计算图，且判别器前向的图会保留到生成器反传。
实测 K=6 / batch=2 在 RTX 4090 (48GB) 上约 29~32 GB，安全。

理论基础（文献）
----------------
见 `tools/train_simvp_rollout.py` 的同名节，此处仅列要点：

* 本脚本同时处理两个问题：
  1. **曝光偏差** → scheduled sampling
     （Bengio et al., 2015, arXiv:1506.03099；思想源自 DAgger，
      Ross et al., 2011, arXiv:1011.0686）
  2. **MSE 最优解 = 后验均值 → 必然模糊** → 对抗损失（感知-失真权衡）
* 与 Bengio 原文的差异：原文为**退火式** schedule，本实现用**固定 p=0.5**。
* 两者的关系：SS 稳住预测的均值/稳定性，GAN 只在稳住的部分上加纹理——
  单用 GAN 会过冲成噪声（锐度比 2.97），单用 SS 会更糊（0.13），
  联合后相对 baseline 两个维度同时更优（RMSE −6.9%，锐度 0.846）。
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
    STD_K, SatcastZarrDataset, _SubsetWithAttrs)
from openstl.datasets.utils import create_loader  # noqa: E402
from openstl.models import SimVP_Model  # noqa: E402
from openstl.utils import load_config  # noqa: E402

MODEL_KEYS = ('in_shape', 'hid_S', 'hid_T', 'N_S', 'N_T', 'model_type', 'mlp_ratio',
              'drop', 'drop_path', 'spatio_kernel_enc', 'spatio_kernel_dec',
              'act_inplace')


# --------------------------------------------------------------- 判别器 ---
class PatchDiscriminator(nn.Module):
    """PatchGAN 风格判别器：输出 patch 级真伪 logits（对纹理/清晰度敏感）。

    输入 (B,1,H,W) -> 输出 (B,1,H/16,W/16)。每个输出位置对应输入的一个感受野块，
    因此能对**局部纹理是否模糊**给出细粒度监督，比单一标量更利于抗模糊。
    """

    def __init__(self, in_ch=1, base=64, n_layers=4, spectral_norm=True):
        super().__init__()

        def conv(a, b, k, s, p, bias):
            c = nn.Conv2d(a, b, k, stride=s, padding=p, bias=bias)
            # 谱归一化：约束判别器 Lipschitz 常数，防止其"用力过猛"把生成器
            # 推向高频噪声（未加谱归一化时曾出现锐度比飙到 2.7 的噪声化）
            return nn.utils.spectral_norm(c) if spectral_norm else c

        layers, ch = [], in_ch
        for i in range(n_layers):
            out = base * min(2 ** i, 8)
            layers += [
                conv(ch, out, 4, 2, 1, bias=(i == 0)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if i > 0:
                layers.append(nn.InstanceNorm2d(out, affine=True))
            ch = out
        layers.append(conv(ch, 1, 4, 1, 1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------ utils ---
def grad_magnitude(x):
    """图像梯度幅值均值（衡量清晰度/锐度）：值越低越模糊、越高越可能是噪声。"""
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    return (dy.mean() + dx.mean()) / 2.0


def rel_grad_loss(pred, real):
    """锐度正则：让预测的梯度幅值**匹配**真实值，量纲归一（理想值 0）。

    与只惩罚"变模糊"不同，这是**双向**的：
      * 预测梯度 < 真实 -> 模糊   -> 产生正损失，推动变清晰
      * 预测梯度 > 真实 -> 噪声   -> 同样产生损失，抑制加噪
    实测未加此项时 GAN 会把锐度比推到 2.7（噪声化），故默认开启。
    """
    gp = grad_magnitude(pred)
    gr = grad_magnitude(real)
    return ((gp - gr) / (gr + 1e-8)) ** 2


def rollout_with_ss(model, x, y, steps, p, step_entry_check=True):
    """滚动 K 步（支持 scheduled sampling），返回 (K,B,C,H,W)。

    Args:
        x: (B,2,C,H,W) 条件帧
        y: (B,K,C,H,W) 每一步的真值（K 步都有，来自 validation 集）
        p: 使用**自己预测**作为下一步输入的概率（0=纯 teacher forcing）
        step_entry_check: 每步入口断言输入恒为 2 帧（防时间维膨胀，见模块 docstring）

    判别器看到的正是本函数返回的 `preds` —— 从第 2 步起它已依赖模型自身预测，
    因此是**滚动预报结果**，而非单步结果。
    """
    cur, outs = x, []
    b = x.shape[0]
    for k in range(steps):
        if step_entry_check and cur.shape[1] != 2:
            raise AssertionError(
                f'rollout 第 {k} 步输入时间维 = {cur.shape[1]}（应为 2）；'
                f'cur={tuple(cur.shape)} x={tuple(x.shape)} y={tuple(y.shape)}')
        pred_t = model(cur)[:, :1]                 # (B,1,C,H,W)：带时间维，供滑动窗口
        outs.append(pred_t.squeeze(1))             # (B,C,H,W)：收集时去掉时间维
        if k < steps - 1:
            gt_next = y[:, k].unsqueeze(1)         # 下一时刻的真值帧 (B,1,C,H,W)
            if p > 0:
                # ⚠️ 条件张量必须是 5 维 (B,1,1,1,1)：4 维会因广播把 batch 维
                # 送到时间轴上，导致 cur 时间维变成 3（B≥2 时必炸）。
                use_own = (torch.rand(b, 1, 1, 1, 1, device=x.device) < p)
                nxt = torch.where(use_own, pred_t.detach(), gt_next)
            else:
                nxt = gt_next
            cur = torch.cat([cur[:, 1:], nxt], dim=1)
    return torch.stack(outs, dim=0)                # (K,B,C,H,W)


def build_model(cfg_path, in_shape):
    cfg = dict(load_config(cfg_path))
    kw = {k: cfg[k] for k in MODEL_KEYS if k in cfg}
    kw['in_shape'] = list(in_shape)
    return SimVP_Model(**kw), kw


def build_loaders(args, K):
    """构建训练/验证 loader。

    多步监督要求 output 有 ≥K 帧真值，只有 validation 集满足（18 帧）。
      * `--val-source validation`：validation 集自身切分训练/验证
      * `--val-source train`：训练集用满**全部** validation 样本，
        验证集改用 train zarr 末尾留出子集（仅 1 帧真值 → 按单步 MSE 验证）
    """
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
    return train_loader, val_loader


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=osp.join(
        _OPENSTL_DIR, 'configs', 'satcast', 'SimVP.py'))
    # 可移植性：数据根优先取环境变量 SATCAST_ROOT
    ap.add_argument('--data-root',
                    default=os.environ.get(
                        'SATCAST_ROOT',
                        '/home/group1/26fall_aiclass/yr/data1/data1/satcast'))
    ap.add_argument('--res-dir', default=osp.join(_OPENSTL_DIR, 'work_dirs'))
    ap.add_argument('--ex-name', default='satcast_simvp_gan_ss')
    ap.add_argument('--train-zarr',
                    default='edm_GOES_ch13_validation_dataset.zarr',
                    help='多步训练数据（该集 output 有 18 帧真值）')
    ap.add_argument('--n-train', type=int, default=1024, help='训练用样本数（取前 N 条）')
    ap.add_argument('--n-val', type=int, default=512, help='验证用样本数（取后 M 条）')
    ap.add_argument('--val-source', choices=['validation', 'train'],
                    default='train',
                    help='验证集来源：validation=从多步训练集中切分；'
                         'train=改用 train zarr 留出子集（训练集可用满全量，默认）')
    ap.add_argument('--rollout-steps', type=int, default=6,
                    help='训练时滚动几步送入判别器')
    ap.add_argument('--sched-samp', type=float, default=0.5,
                    help='scheduled sampling 概率 p（0=纯 teacher forcing）')
    ap.add_argument('--gan-weight', type=float, default=0.05,
                    help='对抗损失权重，**相对于多步 MSE 的比例**'
                         '（0.05 = 对抗项约为 mse 的 5%%）')
    ap.add_argument('--grad-weight', type=float, default=0.05,
                    help='锐度正则权重，**相对于多步 MSE 的比例**；'
                         '双向抑制模糊与加噪，建议保持 >0')
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--lr-d', type=float, default=1e-4, help='判别器学习率')
    ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--init-ckpt', default=None,
                    help='热启动（推荐 baseline 的 best.ckpt）')
    ap.add_argument('--cpu', action='store_true', default=False)
    ap.add_argument('--log-every', type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cpu' if (args.cpu or not torch.cuda.is_available())
                          else 'cuda')
    in_shape = [2, 1, 256, 256]
    K = args.rollout_steps

    # ------------------------- 生成器 -------------------------
    G, kw = build_model(args.config, in_shape)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location='cpu', weights_only=False)
        sd = ck.get('state_dict', ck)
        sd = {k[6:] if k.startswith('model.') else k: v for k, v in sd.items()}
        missing = set(G.state_dict()) - set(sd)
        if missing:
            raise RuntimeError(f'热启动权重不匹配：missing={sorted(missing)[:5]}')
        G.load_state_dict(sd, strict=True)
        print(f'[INIT] 生成器已从 {args.init_ckpt} 热启动')
    G.to(device)

    # ------------------------- 判别器 -------------------------
    D = PatchDiscriminator(in_ch=1, base=64, n_layers=4).to(device)

    opt_g = torch.optim.AdamW(G.parameters(), lr=args.lr, weight_decay=0)
    opt_d = torch.optim.AdamW(D.parameters(), lr=args.lr_d, weight_decay=0)

    train_loader, val_loader = build_loaders(args, K)

    # ⚠️ total_steps 必须用**真实**的 steps/epoch；曾因拍脑袋写常数
    # 导致跑到第 3 轮就 "Tried to step 30001 times" 崩溃。
    steps_per_epoch = len(train_loader)
    sched_g = torch.optim.lr_scheduler.OneCycleLR(
        opt_g, max_lr=args.lr,
        total_steps=max(1, steps_per_epoch * args.epochs), pct_start=0.1)

    save_dir = osp.join(args.res_dir, args.ex_name)
    ck_dir = osp.join(save_dir, 'checkpoints')
    os.makedirs(ck_dir, exist_ok=True)
    with open(osp.join(save_dir, 'gan_ss_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    n_g = sum(p.numel() for p in G.parameters()) / 1e6
    n_d = sum(p.numel() for p in D.parameters()) / 1e6
    print('=' * 74)
    print('SimVP + GAN + scheduled sampling（对抗损失 × 曝光偏差修正）')
    print('=' * 74)
    print(f'[MODEL] {kw}  生成器 {n_g:.3f} M｜判别器 {n_d:.3f} M')
    print(f'[TRAIN] rollout_steps={K}  sched_samp={args.sched_samp}  '
          f'batch={args.batch_size}  epochs={args.epochs}  '
          f'steps/epoch={steps_per_epoch}')
    print('[LOSS ] G = 多步mse * (1 + gan_w*adv + grad_w*rel_grad)，'
          '权重为相对 mse 的比例')
    print('[DISC ] 真实样本 = K 帧真值；假样本 = K 步滚动预报（数量对称）')
    print(f'[OPT  ] lr_G={args.lr}  lr_D={args.lr_d}')
    print(f'[SAVE ] {save_dir}')
    print('=' * 74)

    best_val, history = float('inf'), []
    for epoch in range(args.epochs):
        G.train()
        D.train()
        t0 = time.time()
        acc = {'mse': 0.0, 'adv': 0.0, 'dloss': 0.0, 'rg': 0.0,
               'gm_real': 0.0, 'gm_pred': 0.0}
        nb = 0
        per_step = np.zeros(K)

        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)      # (B,2,C,H,W)
            yb = yb.to(device, non_blocking=True)      # (B,K,C,H,W)

            if epoch == 0 and step == 0:
                print(f'[DEBUG] batch: x={tuple(xb.shape)} y={tuple(yb.shape)}  '
                      f'K={K} p={args.sched_samp}')
                assert xb.shape[1] == 2, (
                    f'输入时间维应为 2（模型按 T=2 构建），实际 {tuple(xb.shape)}')
                assert yb.shape[1] == K, (
                    f'真值时间维应为 K={K}，实际 {tuple(yb.shape)}')

            # ---------- 滚动 K 步（带梯度；第 2 步起依赖自身预测） ----------
            preds = rollout_with_ss(G, xb, yb, K, args.sched_samp)  # (K,B,C,H,W)
            preds_flat = preds.reshape(-1, *preds.shape[2:])        # (K*B,C,H,W)
            real_flat = yb.reshape(-1, *yb.shape[2:])               # (K*B,C,H,W)

            # ---------- 判别器更新 ----------
            # ⚠️ 假样本**必须 detach**：否则 loss_d.backward() 会把 rollout 的计算图
            # 一并释放，紧接着的 loss_g.backward() 就会报
            # "Trying to backward through the graph a second time"。
            # 已实测：若这里漏掉 .detach()，训练第一步就崩。
            d_real = D(real_flat)
            d_fake = D(preds_flat.detach())
            loss_d = (F.mse_loss(d_real, torch.ones_like(d_real))
                      + F.mse_loss(d_fake, torch.zeros_like(d_fake)))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # ---------- 生成器更新 ----------
            multistep_mse = sum(F.mse_loss(preds[k], yb[:, k])
                                for k in range(K)) / K
            d_fake_g = D(preds_flat)
            adv = F.mse_loss(d_fake_g, torch.ones_like(d_fake_g))     # LSGAN, O(1)
            rg = rel_grad_loss(preds_flat, real_flat)                 # O(1)

            # ⚠️ 量纲：multistep_mse≈O(1e-3)，adv/rg≈O(1)，相差约两个数量级。
            # 权重定义为**相对 mse 的比例**，避免对抗项失控（曾把 LT=18 从 11.7 K
            # 恶化到 21.7 K 并产生噪声）。
            loss_g = multistep_mse * (
                1.0 + args.gan_weight * adv + args.grad_weight * rg)

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_g.step()
            sched_g.step()

            with torch.no_grad():
                acc['mse'] += float(multistep_mse)
                acc['adv'] += float(adv)
                acc['dloss'] += float(loss_d)
                acc['rg'] += float(rg)
                acc['gm_real'] += float(grad_magnitude(real_flat))
                acc['gm_pred'] += float(grad_magnitude(preds_flat.detach()))
                for k in range(K):
                    per_step[k] += float(F.mse_loss(preds[k], yb[:, k]))
            nb += 1

            if args.log_every and (step + 1) % args.log_every == 0:
                mem = (torch.cuda.max_memory_allocated() / 2 ** 30
                       if torch.cuda.is_available() else 0.0)
                print(f'  ep{epoch} step{step+1} '
                      f'mse={acc["mse"]/nb:.5f} adv={acc["adv"]/nb:.4f} '
                      f'D={acc["dloss"]/nb:.4f} rg={acc["rg"]/nb:.4f} '
                      f'grad_real={acc["gm_real"]/nb:.5f} '
                      f'grad_pred={acc["gm_pred"]/nb:.5f} mem={mem:.1f}G')

        # ---------- 验证（多步 MSE，不用 SS；kv 动态兼容单帧验证集） ----------
        G.eval()
        vsum, vnb = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                kv = yb.shape[1]
                p = rollout_with_ss(G, xb, yb, kv, 0.0)
                vsum += float(sum(F.mse_loss(p[k], yb[:, k])
                                  for k in range(kv)) / kv)
                vnb += 1
        val = vsum / max(vnb, 1)

        rec = {'epoch': epoch, 'train_mse': acc['mse'] / max(nb, 1),
               'adv': acc['adv'] / max(nb, 1),
               'd_loss': acc['dloss'] / max(nb, 1),
               'rel_grad': acc['rg'] / max(nb, 1),
               'val_multistep_mse': val,
               'val_rmse_K': math.sqrt(val) * STD_K,
               'grad_pred': acc['gm_pred'] / max(nb, 1),
               'grad_real': acc['gm_real'] / max(nb, 1),
               'per_step_train_mse': (per_step / max(nb, 1)).tolist(),
               'time_s': round(time.time() - t0, 1)}
        history.append(rec)
        print(f'[EPOCH {epoch}] val_multistep_mse={val:.5f} '
              f'(rmse {rec["val_rmse_K"]:.3f}K)  '
              f'train_mse={rec["train_mse"]:.5f} adv={rec["adv"]:.4f} '
              f'D={rec["d_loss"]:.4f} '
              f'锐度 pred/real={rec["grad_pred"]/max(rec["grad_real"], 1e-12):.3f}  '
              f'逐步MSE={[round(x, 5) for x in rec["per_step_train_mse"]]}  '
              f'({rec["time_s"]}s)')

        torch.save({'state_dict': G.state_dict(), 'epoch': epoch, 'val_mse': val},
                   osp.join(ck_dir, 'last.pt'))
        if val < best_val:
            best_val = val
            torch.save({'state_dict': G.state_dict(), 'epoch': epoch,
                        'val_mse': val}, osp.join(ck_dir, 'best.pt'))
            # 同时存带 'model.' 前缀的 best.ckpt，供 eval_simvp_baseline.py 加载
            torch.save({'state_dict': {f'model.{k}': v
                                       for k, v in G.state_dict().items()},
                        'epoch': epoch, 'val_mse': val},
                       osp.join(ck_dir, 'best.ckpt'))
            print(f'  -> 新最优 (val={val:.5f})，已同步保存 best.ckpt')
        with open(osp.join(save_dir, 'gan_ss_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print('[DONE] 产物在 ' + save_dir)


if __name__ == '__main__':
    main()
