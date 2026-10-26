#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成**最终产品 A+B** 的 18 步 test 预测，供 `eval_metrics.py` 评估。

流程
----
::

    for k in 1..18:
        â_k = A(cur)                      # A 独立自回归，反馈只用 â
        B_k = B([f_t-10, f_t, â_k], k)    # B 逐步修正，不进入 A 的反馈
        out_k = â_k + B_k                 # 只在末端相加
        cur  = [cur[:,1:], â_k]           # ⚠️ 用 â 而不是 out，保持产品解耦

输出 `(18, N, 1, 256, 256)` **归一化空间**，与 `preds.npy` 完全一致。

三种模式
--------
`--b-ckpt` 给路径   -> 最终产品 A+B
`--b-ckpt none`     -> 只用 A（等价于产品 A 单独，用于对照）
默认                -> 读 `<ex-name>/checkpoints/best.pt`

用法::

    python tools/make_final_preds.py --ex-name productB_k6 \
        --a-ckpt work_dirs/productA_gan_k6/checkpoints/best.ckpt \
        --out work_dirs/productB_k6/final_preds.npy
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

from train_residual_product import ResidualProduct, load_frozen_A  # noqa: E402
from train_simvp_gan import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=osp.join(
        _OPENSTL_DIR, 'configs', 'satcast', 'SimVP.py'))
    ap.add_argument('--data-root',
                    default=os.environ.get(
                        'SATCAST_ROOT',
                        '/home/group1/26fall_aiclass/yr/data1/data1/satcast'))
    ap.add_argument('--ex-name', required=True)
    ap.add_argument('--a-ckpt', required=True, help='产品 A 的 best.ckpt')
    ap.add_argument('--b-ckpt', default=None,
                    help='产品 B 权重；默认取 <ex-name>/checkpoints/best.pt；'
                         '传 none 则只用 A')
    ap.add_argument('--out', default=None)
    ap.add_argument('--steps', type=int, default=18)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--max-samples', type=int, default=None)
    ap.add_argument('--b-leads', type=int, default=None,
                    help='只在 k <= 此值时应用 B，超过则该步用 A 单独。'
                         '默认 None = 全部 steps 都应用（旧行为）。'
                         '依据：B 训练时 K=6，只见过 â_1..â_6 且仅第 1 步有监督；'
                         '对 k>6 的 â_k 属分布外，旧代码还把 lead 钳到 6，'
                         '等于对第 12 步的输入说"你是第 6 步"，信号自相矛盾。')
    ap.add_argument('--b-decay', action='store_true',
                    help='B 的权重从 --b-leads 起线性衰减到 0（需配合 --b-leads），'
                         '比硬截断更平滑')
    ap.add_argument('--cpu', action='store_true', default=False)
    args = ap.parse_args()

    device = torch.device('cpu' if (args.cpu or not torch.cuda.is_available())
                          else 'cuda')
    ex_dir = osp.join(_OPENSTL_DIR, 'work_dirs', 'residual', args.ex_name)
    use_b = (args.b_ckpt or '').lower() != 'none'
    b_ckpt = args.b_ckpt or osp.join(ex_dir, 'checkpoints', 'best.pt')
    if use_b and not osp.exists(b_ckpt):
        raise FileNotFoundError(f'缺产品 B 权重：{b_ckpt}')

    A = load_frozen_A(args.config, args.a_ckpt, [2, 1, 256, 256], device)
    B = None
    if use_b:
        meta = json.load(open(osp.join(ex_dir, 'productB_args.json')))
        B = ResidualProduct(in_ch=3, base=meta.get('width', 48),
                            n_lead=meta.get('rollout_steps', 6)).to(device)
        sd = torch.load(b_ckpt, map_location='cpu', weights_only=False)
        sd = sd.get('state_dict', sd)
        B.load_state_dict(sd, strict=True)
        B.eval()
        print(f'[MODE] 最终产品 = A + B   (B: {b_ckpt})')
    else:
        print('[MODE] 只用产品 A（B=none，作对照）')

    g = zarr.open(osp.join(args.data_root,
                           'edm_GOES_ch13_test_dataset.zarr'), mode='r')
    n_all = g['input_images'].shape[0]
    n = n_all if args.max_samples is None else min(n_all, args.max_samples)
    X = g['input_images']

    out = args.out or osp.join(ex_dir, 'final_preds.npy')
    pred = np.lib.format.open_memmap(
        out, mode='w+', dtype=np.float32,
        shape=(args.steps, n, 1, X.shape[-2], X.shape[-1]))
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, args.batch):
            e = min(s + args.batch, n)
            xb = torch.from_numpy(
                np.asarray(X[s:e], dtype=np.float32)).to(device).unsqueeze(2)
            f_in, f_t = xb[:, 0], xb[:, 1]
            cur = xb
            for k in range(1, args.steps + 1):
                ahat = A(cur)[:, :1].squeeze(1)                  # (B,1,H,W)
                if B is not None:
                    # B 的应用范围：只在 k <= b_leads 时生效。
                    # 训练时 B 只见过 â_1..â_6（K=6）且仅第 1 步有监督，
                    # 对 k>6 属分布外；旧代码还把 lead 钳到 n_lead，
                    # 等于对第 12 步的输入声称"你是第 6 步"，是自相矛盾的信号。
                    apply_b = (args.b_leads is None or k <= args.b_leads)
                    if apply_b:
                        w = 1.0
                        if (args.b_decay and args.b_leads is not None
                                and args.steps > args.b_leads):
                            w = max(0.0, 1.0 - (k - args.b_leads)
                                    / float(args.steps - args.b_leads))
                        xin = torch.cat([f_in, f_t, ahat], dim=1)
                        kk = min(k, B.n_lead)
                        out_k = ahat + w * B(xin, kk)
                    else:
                        out_k = ahat
                else:
                    out_k = ahat
                pred[k - 1, s:e, 0] = out_k.squeeze(1).cpu().numpy()
                # ⚠️ 反馈只用 â（不用 out），保持 A/B 两产品解耦
                cur = torch.cat([cur[:, 1:], ahat.unsqueeze(1)], dim=1)
            if (s // args.batch) % 16 == 0:
                el = time.time() - t0
                print(f'  {e}/{n}  {el:.0f}s  预计 {el*n/max(e,1):.0f}s',
                      flush=True)
    pred.flush()
    print(f'[SAVE] {out}  shape={pred.shape}')
    print('[HINT] python tools/eval_metrics.py '
          f'--preds {out} --out-dir <评估目录> --leads 1,2,3,6,9,12,15,18')


if __name__ == '__main__':
    main()
