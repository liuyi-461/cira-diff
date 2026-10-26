#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把训练好的**残差分支**应用到全 test 集，生成 18 步 rollout 预测。

输出形状 `(18, N, 1, 256, 256)`，**zarr 归一化空间**，与
`eval_simvp_baseline.py` 产出的 `preds.npy` 完全一致，因此可直接喂给
`tools/eval_metrics.py` 做 8 指标评估。

流程（与训练时的反馈结构一致）::

    cur = x                                                  # (B,2,1,H,W)
    for k in 0..17:
        x̂ = 主模型(cur)[:, :1].squeeze(1)                     # 冻结
        base = α·advect(prev, flow) + (1−α)·prev              # flow = 冻结观测流
        out = x̂ + R(concat[x̂, base, f_t])                   # 残差修正
        cur = cat([cur[:,1:], out.unsqueeze(1)])              # 喂修正后的输出

自检（`--check-baseline`）
-------------------------
把 R 的输出层临时置零后重跑同一批样本，与 baseline 的 `preds.npy` 比对。
零初始化下两者应几乎逐位相同（仅有 batch 划分带来的浮点差）——这验证了
「残差分支初始等价于 baseline」以及本脚本 rollout 与既有 baseline 管线的等价性。
若差异过大，说明 rollout 实现有偏差，**不能**开始解释残差分支的效果。

用法::

    python tools/make_residual_preds.py --ex-name residual_R1 \
        --out work_dirs/residual/residual_R1/preds.npy
    python tools/make_residual_preds.py --ex-name residual_R1 --check-baseline
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
import zarr  # noqa: E402

from openstl.datasets.dataloader_satcast import MEAN_K, STD_K  # noqa: E402
from openstl.utils import load_config  # noqa: E402
from train_residual import (MODEL_KEYS, ResidualBranch, advect_batch,  # noqa: E402
                            batch_flow, build_main, build_r_in, norm_from_K)


@torch.no_grad()
def rollout(main, R, xb, alpha, mode, steps, device, flow=None):
    """xb: (B,2,1,H,W) 归一化空间 -> (steps, B, 1, H, W) 归一化空间。"""
    fl = batch_flow(xb) if flow is None else flow
    f_t_norm = xb[:, 1]
    cur = xb
    outs = []
    for k in range(steps):
        xhat = main(cur)[:, :1].squeeze(1)                    # (B,1,H,W)
        prev_K = (cur[:, 1, 0].cpu().numpy() * STD_K + MEAN_K).astype(np.float32)
        base_norm = norm_from_K(advect_batch(prev_K, fl, alpha, k + 1), device)
        out = xhat + R(build_r_in(xhat, base_norm, f_t_norm, mode))
        outs.append(out.unsqueeze(1))                         # (B,1,1,H,W)
        cur = torch.cat([cur[:, 1:], out.unsqueeze(1)], dim=1)
    return torch.cat(outs, dim=1).permute(1, 0, 2, 3, 4)      # (steps,B,1,H,W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=osp.join(
        _OPENSTL_DIR, 'configs', 'satcast', 'SimVP.py'))
    ap.add_argument('--data-root',
                    default=os.environ.get(
                        'SATCAST_ROOT',
                        '/home/group1/26fall_aiclass/yr/data1/data1/satcast'))
    ap.add_argument('--main-ckpt', default=osp.join(
        _OPENSTL_DIR, 'work_dirs', 'satcast_simvp_full', 'checkpoints', 'best.ckpt'))
    ap.add_argument('--ex-name', required=True)
    ap.add_argument('--mode', choices=['R0', 'R1', 'R1sham'],
                    default=None, help='默认从 residual_args.json 读')
    ap.add_argument('--ckpt', default=None,
                    help='残差分支权重；默认取 work_dirs/residual/<ex>/checkpoints/best.pt')
    ap.add_argument('--out', default=None)
    ap.add_argument('--steps', type=int, default=18)
    ap.add_argument('--batch', type=int, default=16,
                    help='⚠️ 必须与产出 baseline preds.npy 时的 batch_size 相同'
                         '（默认 16）。GPU 前向的浮点不可确定性会使每步差 ~0.008 K，'
                         '经 18 步自回归反馈混沌放大到 ~1.1 K（实测 batch 4 vs 16）。'
                         '改变 batch 会让长时效对比出现虚假差异。')
    ap.add_argument('--max-samples', type=int, default=None)
    ap.add_argument('--check-baseline', action='store_true',
                    help='把 R 输出层置零后跑，与 baseline preds.npy 比对（等价性自检）')
    ap.add_argument('--cpu', action='store_true', default=False)
    args = ap.parse_args()

    device = torch.device('cpu' if (args.cpu or not torch.cuda.is_available())
                          else 'cuda')
    ex_dir = osp.join(_OPENSTL_DIR, 'work_dirs', 'residual', args.ex_name)
    mode = args.mode
    alpha = 0.75
    width = 32
    # 两个训练脚本写的 args 文件名不同：train_residual.py -> residual_args.json，
    # train_residual_gan.py -> resgan_args.json。这里都兼容，避免"读不到配置"而崩。
    _args_name = None
    for _cand in ('residual_args.json', 'resgan_args.json'):
        if osp.exists(osp.join(ex_dir, _cand)):
            _args_name = _cand
            break
    if _args_name is None:
        raise SystemExit(f'在 {ex_dir} 下找不到 residual_args.json / resgan_args.json；'
                         f'请用 --mode 显式指定')
    _meta = json.load(open(osp.join(ex_dir, _args_name)))
    if mode is None:
        mode = _meta['mode']
    alpha = _meta.get('alpha', 0.75)
    width = _meta.get('width', 32)
    ckpt = args.ckpt or osp.join(ex_dir, 'checkpoints', 'best.pt')
    for p in (args.main_ckpt, ckpt):
        if not osp.exists(p):
            raise FileNotFoundError(f'缺少权重：{p}')
    out = args.out or osp.join(ex_dir, 'preds.npy')

    main_net = build_main(args.config, [2, 1, 256, 256], args.main_ckpt, device)
    main_net.eval()
    cfg = dict(load_config(args.config))
    kw = {k: cfg[k] for k in MODEL_KEYS if k in cfg}
    kw['in_shape'] = [2, 1, 256, 256]
    from openstl.models import SimVP_Model
    in_ch = 2 if mode == 'R0' else 3
    R = ResidualBranch(in_ch=in_ch, base=width).to(device)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    R.load_state_dict({k: v for k, v in sd.items()}, strict=True)
    R.eval()
    if args.check_baseline:
        torch.nn.init.zeros_(R.out.weight)
        torch.nn.init.zeros_(R.out.bias)
        print('[CHECK] 已把 R 输出层置零 —— 结果应与 baseline preds.npy 几乎相同')

    g = zarr.open(osp.join(args.data_root, 'edm_GOES_ch13_test_dataset.zarr'),
                  mode='r')
    n_all = g['input_images'].shape[0]
    n = n_all if args.max_samples is None else min(n_all, args.max_samples)
    X = g['input_images']

    pred = np.lib.format.open_memmap(
        out, mode='w+', dtype=np.float32, shape=(args.steps, n, 1, 256, 256))
    t0 = time.time()
    for s in range(0, n, args.batch):
        e = min(s + args.batch, n)
        xb = torch.from_numpy(
            np.asarray(X[s:e], dtype=np.float32)).to(device)      # (b,2,H,W)
        xb = xb.unsqueeze(2)                                      # (b,2,1,H,W)
        pred[:, s:e] = rollout(main_net, R, xb, alpha, mode,
                               args.steps, device).cpu().numpy()
        if (s // args.batch) % 16 == 0:
            el = time.time() - t0
            print(f'  {e}/{n}  {el:.0f}s  预计 {el*n/max(e,1):.0f}s', flush=True)
    pred.flush()
    print(f'[SAVE] {out}  shape={pred.shape}')

    if args.check_baseline:
        bp = osp.join(_OPENSTL_DIR, 'work_dirs', 'satcast_simvp_full',
                      'baseline_eval_rollout18_full', 'preds.npy')
        if osp.exists(bp):
            ref = np.load(bp, mmap_mode='r')[:, :n]
            d = np.abs(np.asarray(pred[:, :min(n, 64)], dtype=np.float64)
                       - np.asarray(ref[:, :min(n, 64)], dtype=np.float64))
            print(f'[CHECK] 与 baseline preds.npy 的前 {min(n,64)} 条：'
                  f'max|Δ|={d.max():.3e}  mean|Δ|={d.mean():.3e}（归一化空间）')
            if d.max() < 1e-6:
                print('[CHECK] 判定: ✅ 逐位一致（零初始化下 Δ≡0，且 batch 已对齐）')
            elif d.max() < 1e-3:
                print('[CHECK] 判定: ⚠️ 有微小差（多为 batch 未对齐导致的浮点累积）；'
                      '请把 --batch 设为与 baseline 相同后重试，再解释残差效果。')
            else:
                print('[CHECK] 判定: ❌ 差异显著，rollout 实现或权重有偏差，'
                      '勿解释残差效果。')
        else:
            print(f'[WARN] 未找到 baseline 预测 {bp}，跳过等价性比对')


if __name__ == '__main__':
    main()
