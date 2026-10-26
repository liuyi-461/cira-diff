#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**case 级可预报性分析**：按 RMSE 分类 test case、定误差阈值、找超阈值步数、
并按云系特征析 SimVP 的可预报性。

回答的问题
----------
1. 1024 个 test case 按预报难度怎么分？（用 10min / 1h / 2h / 3h 四个单帧 RMSE）
2. 每个 case 在第几步滚动预报**超过误差阈值**（即"还能信到多久"）？
3. 不同云系（用冷芯面积/云量/空间尺度表征）的可预报性差异？

方法
----
* 逐 case 逐 lead 算 **单帧 RMSE**（亮温 K）：`rmse[c, k]`
* 分类特征：`rmse[:, 0]`(10min) / `[:, 5]`(1h) / `[:, 11]`(2h) / `[:, 17]`(3h)
* 阈值：取**全 case 全 lead 的 RMSE 中位数**与若干分位数，报告多个候选，
  默认用「2h 时效 RMSE 的 75 分位」作为"基本预报能力"的边界（可调）
* 云系特征（只用**输入帧与真值**，不用预报，避免循环论证）：
    - `cold_frac`  冷芯（Tb≤235K）像素占比 —— 云量
    - `grad`       输入帧梯度幅值 —— 纹理/对流活跃度
    - `n_cold`     冷芯像素数的对数 —— 云团规模

用法::

    python tools/case_analysis.py \
        --preds work_dirs/eval_preds/A_cont/preds.npy \
        --tag A_cont --out-dir docs/experiments/EXP-009-case-analysis
"""

import argparse
import json
import os
import os.path as osp
import sys

_HERE = osp.dirname(osp.abspath(__file__))
_OPENSTL_DIR = osp.dirname(_HERE)
for _p in (_OPENSTL_DIR, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import zarr  # noqa: E402

from openstl.datasets.dataloader_satcast import MEAN_K, STD_K  # noqa: E402

COLD = 235.0
KEY_LEADS = [1, 6, 12, 18]          # 10min / 1h / 2h / 3h
KEY_LT = ['10min', '1h', '2h', '3h']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', required=True, help='(18,N,1,H,W) 归一化空间')
    ap.add_argument('--data', default=os.environ.get(
        'SATCAST_ROOT', '/home/group1/26fall_aiclass/yr/data1/data1/satcast')
        + '/edm_GOES_ch13_test_dataset.zarr')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tag', default='arm')
    ap.add_argument('--thresh-quantile', type=float, default=0.75,
                    help='阈值取自该分位数对应的 RMSE（默认 0.75）')
    ap.add_argument('--thresh-lead', type=int, default=12,
                    help='阈值取自哪个时效的单帧 RMSE 分布（默认 12 = 2h）')
    ap.add_argument('--max-samples', type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    P = np.load(args.preds, mmap_mode='r')            # (18,N,1,H,W)
    n_lead, n_all = P.shape[0], P.shape[1]
    n = n_all if args.max_samples is None else min(n_all, args.max_samples)
    g = zarr.open(args.data, mode='r')
    Y = g['output_images']
    X = g['input_images']

    # ---------- 逐 case 逐 lead 单帧 RMSE ----------
    rmse = np.zeros((n, n_lead), dtype=np.float32)
    mae = np.zeros((n, n_lead), dtype=np.float32)
    for s in range(0, n, 64):
        e = min(s + 64, n)
        p = np.asarray(P[:, s:e, 0], dtype=np.float64) * STD_K + MEAN_K  # (18,b,H,W)
        t = np.asarray(Y[s:e], dtype=np.float64) * STD_K + MEAN_K        # (b,18,H,W)
        p = np.transpose(p, (1, 0, 2, 3))                                # (b,18,H,W)
        d = p - t
        rmse[s:e] = np.sqrt((d ** 2).mean(axis=(2, 3)))
        mae[s:e] = np.abs(d).mean(axis=(2, 3))

    # ---------- 云系特征（只用输入帧与真值） ----------
    cold_frac = np.zeros(n, dtype=np.float32)
    grad_in = np.zeros(n, dtype=np.float32)
    n_cold = np.zeros(n, dtype=np.float32)
    for s in range(0, n, 64):
        e = min(s + 64, n)
        x = np.asarray(X[s:e], dtype=np.float64) * STD_K + MEAN_K   # (b,2,H,W)
        t = np.asarray(Y[s:e], dtype=np.float64) * STD_K + MEAN_K
        cold_frac[s:e] = (t <= COLD).mean(axis=(1, 2, 3))
        gy = np.abs(x[:, 1, 1:] - x[:, 1, :-1]).mean(axis=(1, 2))
        gx = np.abs(x[:, 1, :, 1:] - x[:, 1, :, :-1]).mean(axis=(1, 2))
        grad_in[s:e] = (gy + gx) / 2
        n_cold[s:e] = np.log1p((t <= COLD).sum(axis=(1, 2, 3)))

    # ---------- 阈值与超阈值步数 ----------
    thresh_pool = float(np.quantile(rmse[:, args.thresh_lead - 1],
                                    args.thresh_quantile))
    exceed_step = np.full(n, -1, dtype=np.int32)     # -1 = 从未超阈值
    for c in range(n):
        w = np.nonzero(rmse[c] > thresh_pool)[0]
        if len(w):
            exceed_step[c] = int(w[0]) + 1            # 1-based

    # ---------- 难度分类（按 4 个关键时效的 RMSE 均值分三档） ----------
    key_mean = rmse[:, [k - 1 for k in KEY_LEADS]].mean(axis=1)
    q1, q2 = np.quantile(key_mean, [1 / 3, 2 / 3])
    tier = np.digitize(key_mean, [q1, q2])           # 0=易 1=中 2=难

    # ---------- 云系分桶 ----------
    cf_q = np.quantile(cold_frac, [0.33, 0.67])
    cloud_bin = np.digitize(cold_frac, cf_q)         # 0=少云 1=中 2=多云

    # ---------- 汇总 ----------
    def _stat(v):
        v = np.asarray(v, dtype=np.float64)
        return {'mean': float(v.mean()), 'std': float(v.std()),
                'median': float(np.median(v)),
                'p25': float(np.quantile(v, .25)), 'p75': float(np.quantile(v, .75))}

    out = {
        'meta': {'preds': osp.abspath(args.preds), 'tag': args.tag,
                 'n_cases': int(n), 'n_lead': int(n_lead),
                 'cold_thresh_K': COLD, 'key_leads': KEY_LT,
                 'thresh_pool_K': thresh_pool,
                 'thresh_def': f'{KEY_LT[KEY_LEADS.index(args.thresh_lead)]}时效 '
                               f'RMSE 的 {args.thresh_quantile:.0%} 分位'},
        'rmse_by_case': rmse.astype(np.float32).tolist(),
        'mae_by_case': mae.astype(np.float32).tolist(),
        'case_features': {
            'cold_frac': cold_frac.tolist(),
            'grad_input': grad_in.tolist(),
            'log_n_cold': n_cold.tolist(),
        },
        'per_case': {
            'exceed_step': exceed_step.tolist(),
            'tier': tier.tolist(),
            'cloud_bin': cloud_bin.tolist(),
            'key_mean_rmse': key_mean.tolist(),
        },
        'tier_stats': {},
        'cloud_bin_stats': {},
        'exceed_step_dist': {},
    }
    for name, arr in [('easy(易)', 0), ('medium(中)', 1), ('hard(难)', 2)]:
        m = tier == arr
        if m.sum() == 0:
            continue
        out['tier_stats'][name] = {
            'n': int(m.sum()),
            'rmse_at': {KEY_LT[i]: _stat(rmse[m, k - 1])
                        for i, k in enumerate(KEY_LEADS)},
            'mean_exceed_step': float(exceed_step[m].mean())
            if (exceed_step[m] >= 0).any() else None,
            'cold_frac': _stat(cold_frac[m]),
            'grad_input': _stat(grad_in[m]),
        }
    for name, arr in [('少云', 0), ('中', 1), ('多云', 2)]:
        m = cloud_bin == arr
        if m.sum() == 0:
            continue
        out['cloud_bin_stats'][name] = {
            # ⚠️ 必须记 bin_index：dict 只插入有数据的桶，插入顺序与桶索引
            #    不再一一对应（如只有 bin1/bin2 有数据时，enumerate 会把
            #    bin1 误标成第 0 个名字）。绘图端靠这个字段取回正确索引。
            'bin_index': int(arr),
            'n': int(m.sum()),
            'cold_frac_range': [float(cold_frac[m].min()), float(cold_frac[m].max())],
            'rmse_at': {KEY_LT[i]: _stat(rmse[m, k - 1])
                        for i, k in enumerate(KEY_LEADS)},
            'mean_exceed_step': float(exceed_step[m].mean())
            if (exceed_step[m] >= 0).any() else None,
        }
    vals, cnts = np.unique(exceed_step[exceed_step >= 0], return_counts=True)
    out['exceed_step_dist'] = {str(int(v)): int(c) for v, c in zip(vals, cnts)}
    out['n_never_exceed'] = int((exceed_step < 0).sum())

    with open(osp.join(args.out_dir, f'case_analysis_{args.tag}.json'), 'w') as f:
        json.dump(out, f)

    # ---------- 打印 ----------
    print('=' * 92)
    print(f'case 级可预报性分析（{args.tag}，{n} 个 test case）')
    print('=' * 92)
    print(f'阈值定义：{out["meta"]["thresh_def"]} -> **{thresh_pool:.3f} K**')
    print(f'从未超阈值的 case 数：{out["n_never_exceed"]} / {n}'
          f'（{out["n_never_exceed"]/n*100:.1f}%）')
    print()
    print('--- 难度分档（按 10min/1h/2h/3h 四个单帧 RMSE 的均值三分位） ---')
    print(f'{"档位":<12}{"case数":>7}{"10min":>9}{"1h":>9}{"2h":>9}{"3h":>9}'
          f'{"平均超阈步":>11}{"云量":>8}{"输入梯度":>9}')
    for name, v in out['tier_stats'].items():
        r = v['rmse_at']
        print(f'{name:<12}{v["n"]:>7}{r["10min"]["mean"]:>9.2f}{r["1h"]["mean"]:>9.2f}'
              f'{r["2h"]["mean"]:>9.2f}{r["3h"]["mean"]:>9.2f}'
              f'{(v["mean_exceed_step"] or -1):>11.1f}{v["cold_frac"]["mean"]:>8.3f}'
              f'{v["grad_input"]["mean"]:>9.4f}')
    print()
    print('--- 按云量分桶（真值冷芯占比三分位） ---')
    print(f'{"云量":<8}{"case数":>7}{"冷芯占比":>14}{"10min":>9}{"1h":>9}{"2h":>9}{"3h":>9}{"平均超阈步":>11}')
    for name, v in out['cloud_bin_stats'].items():
        r = v['rmse_at']
        print(f'{name:<8}{v["n"]:>7}{v["cold_frac_range"][0]:>7.3f}~{v["cold_frac_range"][1]:<6.3f}'
              f'{r["10min"]["mean"]:>9.2f}{r["1h"]["mean"]:>9.2f}{r["2h"]["mean"]:>9.2f}'
              f'{r["3h"]["mean"]:>9.2f}{(v["mean_exceed_step"] or -1):>11.1f}')
    print()
    print('--- 超阈值步数分布（第几步首次超过阈值） ---')
    for k in sorted(out['exceed_step_dist'], key=lambda x: int(x)):
        c = out['exceed_step_dist'][k]
        print(f'  第 {k:>2} 步（{int(k)*10:>3}min）：{c:>4} 个 case'
              f'（{c/n*100:>5.1f}%）')
    print('=' * 92)
    print(f'[SAVE] {osp.join(args.out_dir, f"case_analysis_{args.tag}.json")}')


if __name__ == '__main__':
    main()
