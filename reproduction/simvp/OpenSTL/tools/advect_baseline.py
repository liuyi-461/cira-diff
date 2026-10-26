#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**光流平流外推基线**：两种模式，用于公平归因"位移误差占多少"。

为什么需要这个脚本
------------------
此前所有对比都以 **persistence（上一帧不动）** 为参照，但 dcm/wind_residual 的
阶段 A 已显示：光流平流比 persistence 好约 18.7%。只用 persistence 当基线会
**高估神经模型的贡献**，也无法回答"位移误差在总误差中占多少"。

两种模式（关键区别在**有无误差反馈回路**）
---------------------------------------
| 模式 | 源帧 | 光流 | 外推 | 有无误差反馈 |
|---|---|---|---|---|
| `frozen` | 最后一帧**观测** f1 | 输入两帧估得，**全程复用** | k × flow 累积 | **无** |
| `selfconsistent` | 上一步**预测** | 每步用上两步**重估** | 1 × flow | **有** |

`frozen` 与 dcm 的阶段 A 口径一致（可比）；`selfconsistent` 才是与神经 rollout
**结构对等**的基线——神经模型第 k 步的输入是 `[pred_{k-2}, pred_{k-1}]`，
自一致平流也每步用最新的两帧重估流，因此同样会累积并放大自身误差。
**只有 selfconsistent 才能公平回答"纯位移策略能做到多好"。**

⚠️ 引用注意（来自对 dcm 的批判性复核，见 forecast-realism-report.md §6/§8）
* dcm 的 −16.2% 是对着 `frozen` 基线测的，而 frozen 无误差反馈、长时效会
  累积弧状伪影 —— 引用该数字必须加这条 caveat。
* α **在 validation 上选**，不在 test 上选（dcm 是在 test 上扫描的，基线偏乐观）。

数据契约
--------
* 输入：任一 zarr（train 只有 1 帧真值，故 18 步外推只能用 validation/test）
* 输出：`(steps, N, 1, 256, 256)` **归一化空间**，与 `preds.npy` 完全一致，
  可直接喂给 `tools/eval_metrics.py`
* 全部计算在**亮温 K** 空间做（光流与 warp 都需要物理量级），最后转回归一化

用法::

    # ① 在 validation 上扫 α（两种模式各自选最优）
    python tools/advect_baseline.py --data <validation.zarr> --scan-alpha \
        --scan-samples 1024 --out-dir work_dirs/advect_baseline/val_scan

    # ② 用选定 α 在 test 上生成两种模式的预测
    python tools/advect_baseline.py --data <test.zarr> --mode both --alpha 0.75 \
        --out-dir work_dirs/advect_baseline/test

    # ③ 评估（与其它臂同口径）
    python tools/eval_metrics.py \
        --preds work_dirs/advect_baseline/test/selfconsistent_a0.75/preds.npy \
        --out-dir docs/experiments/EXP-008-metrics-baseline/adv_sc_a075/
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
import zarr  # noqa: E402

# 复用已通过符号自检的光流/warp 实现（勿重复实现，避免符号约定漂移）
from eval_metrics import MEAN_K, STD_K, flow_dis, warp  # noqa: E402


# ------------------------------------------------------------------ 滚动 ---
def rollout_frozen(f0, f1, steps, alpha, flow=None):
    """冻结流平流：源=最后一帧观测 f1，流复用，k×flow 累积外推。

    与 dcm 阶段 A 口径一致（`advect(f1, fl, load_step*k)`），可横向比。
    **无误差反馈**：不会累积放大自身错误。
    """
    if flow is None:
        flow = flow_dis(f0, f1)
    out = np.empty((steps,) + f1.shape, dtype=np.float32)
    for k in range(1, steps + 1):
        out[k - 1] = alpha * warp(f1, flow, k) + (1.0 - alpha) * f1
    return out, flow


def rollout_selfconsistent(f0, f1, steps, alpha):
    """自一致平流：每步用上两步**重估**光流，外推上一步预测。

    与神经 rollout 结构对等（第 k 步输入 = `[pred_{k-2}, pred_{k-1}]`），
    **有误差反馈**：自身误差会进入下一帧的源与流估计并被放大。
    """
    prev, cur = np.asarray(f0, dtype=np.float64), np.asarray(f1, dtype=np.float64)
    out = np.empty((steps,) + f1.shape, dtype=np.float32)
    for k in range(1, steps + 1):
        fl = flow_dis(prev, cur)                       # 每步重估（用上两步）
        nxt = alpha * warp(cur, fl, 1) + (1.0 - alpha) * cur
        out[k - 1] = nxt
        prev, cur = cur, nxt
    return out


# ------------------------------------------------------------------ 工具 ---
def _to_norm(k_arr):
    """亮温 K -> 归一化空间（与 preds.npy 一致）。"""
    return ((np.asarray(k_arr, dtype=np.float32) - MEAN_K) / STD_K).astype(np.float32)


def pooled_mae(pred_norm, truth_norm):
    """18 时效池化 MAE（亮温 K）—— 用于在 validation 上选 α。"""
    d = np.abs(pred_norm.astype(np.float64) - truth_norm.astype(np.float64))
    return float(d.mean()) * STD_K


def run_split(zarr_path, mode, alpha, steps, max_samples, out_path, log_tag=''):
    """对某个 zarr 跑一种模式，返回 (preds_norm (steps,N,1,H,W), pooled_mae_K)。

    `truth` 仅用于在 validation 上算 MAE 选 α；test 上传 None 则不算。
    """
    g = zarr.open(zarr_path, mode='r')
    n_all = g['input_images'].shape[0]
    n = n_all if not max_samples else min(n_all, max_samples)
    X = g['input_images']                       # (N,2,H,W) 归一化
    has_truth = 'output_images' in g and g['output_images'].shape[1] >= steps
    Y = g['output_images'] if has_truth else None

    pred = np.lib.format.open_memmap(
        out_path, mode='w+', dtype=np.float32,
        shape=(steps, n, 1, X.shape[-2], X.shape[-1]))
    t0 = time.time()
    for s in range(0, n, 16):
        e = min(s + 16, n)
        xb = np.asarray(X[s:e], dtype=np.float32)          # (b,2,H,W) 归一化
        kb = xb * STD_K + MEAN_K                           # -> K
        for i in range(e - s):
            f0, f1 = kb[i, 0], kb[i, 1]
            if mode == 'frozen':
                pk, _ = rollout_frozen(f0, f1, steps, alpha)
            else:
                pk = rollout_selfconsistent(f0, f1, steps, alpha)
            pred[:, s + i, 0] = _to_norm(pk)
        if (s // 16) % 32 == 0:
            el = time.time() - t0
            print(f'  [{log_tag}] {e}/{n}  {el:.0f}s  预计 {el*n/max(e,1):.0f}s',
                  flush=True)
    pred.flush()

    mae = None
    if Y is not None:
        acc, cnt = 0.0, 0
        for s in range(0, n, 64):
            e = min(s + 64, n)
            yb = np.asarray(Y[s:e, :steps], dtype=np.float32)      # (b,steps,H,W)
            pb = np.asarray(pred[:, s:e, 0], dtype=np.float32)
            pb = np.transpose(pb, (1, 0, 2, 3))                    # (b,steps,H,W)
            acc += float(np.abs(pb - yb).sum())
            cnt += yb.size
        mae = acc / cnt * STD_K
    return pred, mae


def scan_alpha(zarr_path, mode, alphas, steps, max_samples, work_dir):
    """在某个 split（应选 validation）上扫 α，返回 {alpha: pooled_mae_K}。"""
    res = {}
    for a in alphas:
        tmp = osp.join(work_dir, f'_scan_{mode}_a{a}.npy')
        _, mae = run_split(zarr_path, mode, a, steps, max_samples, tmp,
                           log_tag=f'scan {mode} α={a}')
        if mae is not None:
            res[a] = mae
            print(f'  [SCAN] mode={mode:<15} α={a:<5} 池化 MAE = {mae:.4f} K',
                  flush=True)
        if osp.exists(tmp):
            os.remove(tmp)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='zarr 路径（validation 或 test）')
    ap.add_argument('--mode', choices=['frozen', 'selfconsistent', 'both'],
                    default='both')
    ap.add_argument('--alpha', type=float, default=0.75,
                    help='平流混合系数（1=纯平流）。扫 α 时忽略')
    ap.add_argument('--scan-alpha', action='store_true',
                    help='在 --data 上扫 α 并选最优（应传 validation）')
    ap.add_argument('--alphas', default='0,0.25,0.5,0.75,1.0')
    ap.add_argument('--scan-samples', type=int, default=1024,
                    help='扫 α 用的样本数（selfconsistent 每 α 都要重跑，'
                         '可用子集加速；正式出预测用全量）')
    ap.add_argument('--steps', type=int, default=18)
    ap.add_argument('--max-samples', type=int, default=None)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--selftest', action='store_true',
                    help='只跑光流符号自检后退出')
    args = ap.parse_args()

    if args.selftest:
        from eval_metrics import selftest_flow_sign
        selftest_flow_sign()
        return

    os.makedirs(args.out_dir, exist_ok=True)
    modes = (['frozen', 'selfconsistent'] if args.mode == 'both'
             else [args.mode])
    alphas = [float(x) for x in args.alphas.split(',') if x.strip()]

    summary = {'data': osp.abspath(args.data), 'steps': args.steps,
               'modes': {}, 'alphas_scanned': alphas}

    # ---------- ① 扫 α（选 validation） ----------
    chosen = {}
    if args.scan_alpha:
        print(f'=== 在 {osp.basename(args.data)} 上扫 α'
              f'（{args.scan_samples} 条）===')
        for m in modes:
            r = scan_alpha(args.data, m, alphas, args.steps,
                           args.scan_samples, args.out_dir)
            summary['modes'][m] = {'alpha_scan_mae_K': r}
            if r:
                best = min(r, key=r.get)
                chosen[m] = best
                print(f'  => {m}: 选 α={best}（MAE {r[best]:.4f} K）')
        summary['chosen_alpha'] = chosen
        with open(osp.join(args.out_dir, 'alpha_scan.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'[SAVE] {osp.join(args.out_dir, "alpha_scan.json")}')
        if not args.alpha:
            return

    # ---------- ② 用选定 α 出预测 ----------
    for m in modes:
        a = chosen.get(m, args.alpha)
        sub = osp.join(args.out_dir, f'{m}_a{a:g}')
        os.makedirs(sub, exist_ok=True)
        out_npy = osp.join(sub, 'preds.npy')
        print(f'=== 生成 {m}（α={a:g}，{args.steps} 步）-> {out_npy} ===')
        pred, mae = run_split(args.data, m, a, args.steps, args.max_samples,
                              out_npy, log_tag=m)
        rec = {'mode': m, 'alpha': a, 'steps': args.steps,
               'n_samples': int(pred.shape[1]), 'preds': osp.abspath(out_npy),
               'pooled_mae_K': mae}
        summary['modes'].setdefault(m, {}).update(rec)
        with open(osp.join(sub, 'advect_meta.json'), 'w') as f:
            json.dump(rec, f, indent=2)
        if mae is not None:
            print(f'  [MAE] {m} α={a:g} 池化 MAE = {mae:.4f} K')

    with open(osp.join(args.out_dir, 'advect_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[DONE] 汇总见 {osp.join(args.out_dir, "advect_summary.json")}')
    print('[HINT] 评估: python tools/eval_metrics.py '
          f'--preds {args.out_dir}/<mode>_a<α>/preds.npy '
          f'--out-dir docs/experiments/EXP-008-metrics-baseline/<arm>/')


if __name__ == '__main__':
    main()
