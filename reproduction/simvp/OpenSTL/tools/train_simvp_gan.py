#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SimVP + **GAN loss** 的滚动预报训练（解决长时效"越推越模糊"）。

动机
----
纯 MSE + teacher forcing 训练出的 SimVP，单步（10 min）清晰，但自回归滚动多步后
预测逐渐趋向**均值场 / 模糊**。原因是 MSE 对多个合理未来取平均时最优解是模糊的。

做法（按你的思路）
----------------
1. 训练时**先滚动预报 K 步**（不再是只做单步），把**滚动结果**送入判别器 D；
2. D 判断一帧是否为**真实**云图：真实（清晰）→ loss 低；预报（模糊）→ loss 高；
3. 生成端 G 的对抗损失约束它产生**越来越真实（不模糊）**的结果。

⚠️ 关键约束与应对
----------------
**train 集只有 1 帧真值**（`output_images` 形状 `(N,1,256,256)`），因此**无法**计算
多步监督损失（没有 t+20min、t+30min 的真值）。
但**判别器不需要真值配对**——只判断"像不像真实云图"。所以：
  * 真实样本：训练批里的真实帧（2 帧输入 + 1 帧真值，共 3 帧/样本）
  * 假样本：滚动 K 步的预报帧
这样在只有单帧真值的条件下依然成立。
（长时效的**有真值评估**仍在 test 集做——test 的 `output_images` 有 18 帧。）

损失
----
* 监督：`MSE(pred@step1, gt@step1)` —— 保住 10 分钟精度
* 对抗（LSGAN，比 BCE 稳定）：
  * D: `mean((D(real)-1)^2) + mean((D(fake.detach())-0)^2)`
  * G: `lambda_gan * mean((D(fake)-1)^2)`
* 可选锐度正则 `--grad-weight`：预测与真值的**梯度幅值**差异（直接惩罚"变模糊"）

锐度监控
--------
每轮记录 `grad_mean_real` 与 `grad_mean_pred`（图像梯度幅值均值）。
若 pred 明显低于 real 即说明模糊；GAN 训练后两者应接近。

用法::
    python tools/train_simvp_gan.py --rollout-steps 3 --gan-weight 0.01

⚠️ 显存：滚动 K 步会构建 K 倍计算图。默认 `batch_size=4, rollout-steps=3`；
若 OOM 请减小 `--batch-size` 或用 `--detach-steps`（只对最后一步回传梯度）。
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

from openstl.datasets.dataloader_satcast import load_data  # noqa: E402
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


def rollout(model, x, steps, detach_intermediate=False):
    """滚动预报 K 步。x: (B, T=2, C, H, W) -> 返回 (K, B, C, H, W)。

    注意形状：``model(cur)`` 输出 (B, pre_seq_length=2, C, H, W)，
    ``[:, :1]`` 后是 (B, **1**, C, H, W)——仍带时间维。滚动窗口需要这个时间维，
    但收集结果时要 squeeze 掉，否则 stack 后会多出一维。
    """
    cur, outs = x, []
    for k in range(steps):
        if detach_intermediate and k > 0:
            cur = cur.detach()
        pred_t = model(cur)[:, :1]        # (B, 1, C, H, W)：带时间维，供滑动窗口
        outs.append(pred_t.squeeze(1))    # (B, C, H, W)：供收集
        cur = torch.cat([cur[:, 1:], pred_t], dim=1)
    return torch.stack(outs, dim=0)       # (K, B, C, H, W)


def build_model(cfg_path, in_shape):
    cfg = dict(load_config(cfg_path))
    kw = {k: cfg[k] for k in MODEL_KEYS if k in cfg}
    kw['in_shape'] = list(in_shape)
    return SimVP_Model(**kw), kw


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=osp.join(
        _OPENSTL_DIR, 'configs', 'satcast', 'SimVP.py'))
    # 可移植性：数据根优先取环境变量 SATCAST_ROOT
    ap.add_argument('--data-root',
                    default=os.environ.get('SATCAST_ROOT',
                                           '/home/group1/26fall_aiclass/yr/data1/data1/satcast'))
    ap.add_argument('--res-dir', default=osp.join(_OPENSTL_DIR, 'work_dirs'))
    ap.add_argument('--ex-name', default='satcast_simvp_gan')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--lr-d', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rollout-steps', type=int, default=3,
                    help='训练时滚动几步送入判别器')
    ap.add_argument('--gan-weight', type=float, default=0.05,
                    help='对抗损失权重，**相对于 MSE 的比例**（0.05 = 对抗项约为 mse 的 5%%）')
    ap.add_argument('--grad-weight', type=float, default=0.05,
                    help='锐度正则权重，**相对于 MSE 的比例**；双向抑制模糊与加噪，'
                         '建议保持 >0')
    ap.add_argument('--detach-steps', action='store_true', default=False,
                    help='只对最后一步回传梯度（省显存）')
    ap.add_argument('--limit-train', type=int, default=None)
    ap.add_argument('--limit-val', type=int, default=None)
    ap.add_argument('--limit-test', type=int, default=None)
    ap.add_argument('--init-ckpt', default=None,
                    help='可选：从 baseline 的 best.ckpt 热启动生成器')
    ap.add_argument('--cpu', action='store_true', default=False)
    ap.add_argument('--log-every', type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cpu' if (args.cpu or not torch.cuda.is_available()) else 'cuda')
    in_shape = [2, 1, 256, 256]

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

    D = PatchDiscriminator(in_ch=1, base=64, n_layers=4).to(device)

    opt_g = torch.optim.AdamW(G.parameters(), lr=args.lr, weight_decay=0)
    opt_d = torch.optim.AdamW(D.parameters(), lr=args.lr_d, weight_decay=0)

    train_loader, val_loader, _ = load_data(
        batch_size=args.batch_size, val_batch_size=8,
        data_root=args.data_root, num_workers=args.num_workers,
        limit_train=args.limit_train, limit_val=args.limit_val,
        limit_test=args.limit_test)

    # ⚠️ total_steps 必须用**真实**的 steps/epoch（len(train_loader)），
    # 不能拍脑袋写 epochs*1000：真实值为 35595/4 ≈ 8899，
    # 曾因此跑到第 3 轮就 "Tried to step 30001 times" 崩溃。
    steps_per_epoch = len(train_loader)
    sched_g = torch.optim.lr_scheduler.OneCycleLR(
        opt_g, max_lr=args.lr,
        total_steps=max(1, steps_per_epoch * args.epochs), pct_start=0.1)

    save_dir = osp.join(args.res_dir, args.ex_name)
    ck_dir = osp.join(save_dir, 'checkpoints')
    os.makedirs(ck_dir, exist_ok=True)

    with open(osp.join(save_dir, 'gan_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print('=' * 74)
    print('SimVP + GAN（滚动预报抗模糊训练）')
    print('=' * 74)
    print(f'[MODEL] {kw}  参数 {sum(p.numel() for p in G.parameters())/1e6:.3f} M')
    print(f'[DISC ] PatchDiscriminator  参数 '
          f'{sum(p.numel() for p in D.parameters())/1e6:.3f} M')
    print(f'[GAN  ] rollout_steps={args.rollout_steps}  '
          f'gan_weight={args.gan_weight}  grad_weight={args.grad_weight}  '
          f'detach_steps={args.detach_steps}')
    print(f'[OPT  ] lr_G={args.lr}  lr_D={args.lr_d}  batch={args.batch_size}  '
          f'epochs={args.epochs}  steps/epoch={steps_per_epoch}')
    print('[LOSS ] G = mse * (1 + gan_w*adv + grad_w*rel_grad)，权重为相对 mse 的比例')
    print(f'[SAVE ] {save_dir}')
    print('=' * 74)

    best_val = float('inf')
    history = []

    for epoch in range(args.epochs):
        G.train()
        D.train()
        t0 = time.time()
        acc = {'mse': 0.0, 'adv': 0.0, 'dloss': 0.0, 'gm_real': 0.0,
               'gm_pred': 0.0, 'grad': 0.0}
        nb = 0

        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)      # (B,2,1,H,W)
            yb = yb.to(device, non_blocking=True)      # (B,1,1,H,W)

            # ---------- 滚动 K 步（带梯度） ----------
            preds = rollout(G, xb, args.rollout_steps, args.detach_steps)  # (K,B,1,H,W)

            # ---------- 判别器更新 ----------
            # 真实帧：本批的 2 帧输入 + 1 帧真值（都是真实观测）
            real = torch.cat([xb[:, 0], xb[:, 1], yb[:, 0]], dim=0)        # (3B,1,H,W)
            fake = preds.detach().reshape(-1, *preds.shape[2:])             # (K*B,1,H,W)
            d_real = D(real)
            d_fake = D(fake)
            loss_d = (F.mse_loss(d_real, torch.ones_like(d_real))
                      + F.mse_loss(d_fake, torch.zeros_like(d_fake)))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # ---------- 生成器更新 ----------
            mse = F.mse_loss(preds[0], yb[:, 0])        # 只第 1 步有真值
            d_fake_g = D(preds.reshape(-1, *preds.shape[2:]))
            adv = F.mse_loss(d_fake_g, torch.ones_like(d_fake_g))     # O(1)
            rg = rel_grad_loss(preds.reshape(-1, *preds.shape[2:]), real)  # O(1)

            # ⚠️ 量纲是关键：mse 约 6.6e-3，而 adv / rg 都是 O(1)，相差约两个数量级。
            # 若写成 mse + w*adv，即使 w=0.01 也会让对抗项 ≈5e-3，与 mse 同量级，
            # 直接把模型带偏（实测：LT=18 RMSE 从 11.7 K 恶化到 21.7 K，并产生噪声）。
            # 因此权重定义为**相对 mse 的比例**：gan_weight=0.05 即"对抗项约为 mse 的 5%"，
            # 语义清晰且不会失控。
            loss_g = mse * (1.0 + args.gan_weight * adv + args.grad_weight * rg)

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_g.step()
            sched_g.step()

            with torch.no_grad():
                acc['mse'] += float(mse)
                acc['adv'] += float(adv)
                acc['dloss'] += float(loss_d)
                acc['gm_real'] += float(grad_magnitude(real))
                acc['gm_pred'] += float(
                    grad_magnitude(preds.detach().reshape(-1, *preds.shape[2:])))
                acc['grad'] += float(rg)
            nb += 1

            if args.log_every and (step + 1) % args.log_every == 0:
                print(f'  ep{epoch} step{step+1} '
                      f'mse={acc["mse"]/nb:.5f} adv={acc["adv"]/nb:.4f} '
                      f'D={acc["dloss"]/nb:.4f} '
                      f'grad_real={acc["gm_real"]/nb:.5f} '
                      f'grad_pred={acc["gm_pred"]/nb:.5f} '
                      f'mem={torch.cuda.max_memory_allocated()/2**30:.1f}G')

        # ---------- 验证（单步 MSE + 锐度） ----------
        G.eval()
        vsum, vgrad_pred, vgrad_real, vnb = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                # 必须 squeeze(1)：G(xb)[:, :1] 是 (B,1,C,H,W) 仍带时间维，
                # 直接与 (B,C,H,W) 的真值比较会触发广播（val_mse 虚高），
                # 且 grad_magnitude 在 5 维上会取到空张量 -> NaN。
                p = G(xb)[:, :1].squeeze(1)            # (B,C,H,W)
                vsum += float(F.mse_loss(p, yb[:, 0]))
                vgrad_pred += float(grad_magnitude(p))
                vgrad_real += float(grad_magnitude(yb[:, 0]))
                vnb += 1
        val = vsum / max(vnb, 1)
        gp = vgrad_pred / max(vnb, 1)
        gr = vgrad_real / max(vnb, 1)

        rec = {
            'epoch': epoch, 'train_mse': acc['mse'] / max(nb, 1),
            'adv': acc['adv'] / max(nb, 1), 'd_loss': acc['dloss'] / max(nb, 1),
            'val_mse': val, 'val_rmse_K': math.sqrt(val) * 19.32967519050003,
            'grad_pred': gp, 'grad_real': gr,
            'sharpness_ratio': gp / gr if gr > 0 else None,
            'time_s': round(time.time() - t0, 1),
        }
        history.append(rec)
        print(f'[EPOCH {epoch}] val_mse={val:.5f} rmse={rec["val_rmse_K"]:.3f}K '
              f'train_mse={rec["train_mse"]:.5f} adv={rec["adv"]:.4f} '
              f'D={rec["d_loss"]:.4f} '
              f'锐度 pred/real={rec["sharpness_ratio"]:.3f} '
              f'({rec["time_s"]}s)')

        torch.save({'state_dict': G.state_dict(), 'epoch': epoch,
                    'val_mse': val}, osp.join(ck_dir, 'last.pt'))
        if val < best_val:
            best_val = val
            torch.save({'state_dict': G.state_dict(), 'epoch': epoch,
                        'val_mse': val}, osp.join(ck_dir, 'best.pt'))
            # 同时存一份带 'model.' 前缀的 best.ckpt，
            # 使 tools/eval_simvp_baseline.py 能直接加载评估（与 baseline 一致）
            torch.save({'state_dict': {f'model.{k}': v
                                       for k, v in G.state_dict().items()},
                        'epoch': epoch, 'val_mse': val},
                       osp.join(ck_dir, 'best.ckpt'))
            print(f'  -> 新最优 (val_mse={val:.5f})，已同步保存 best.ckpt')

        with open(osp.join(save_dir, 'gan_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    # 同时保存成与 baseline 一致的 ckpt 命名，便于 eval 脚本直接加载
    torch.save({'state_dict': {f'model.{k}': v for k, v in G.state_dict().items()},
                'epoch': args.epochs}, osp.join(ck_dir, 'best.ckpt'))
    print('[DONE] 训练完成，产物在 ' + save_dir)


if __name__ == '__main__':
    main()
