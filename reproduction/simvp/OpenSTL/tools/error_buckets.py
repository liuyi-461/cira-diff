#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""误差的**分桶诊断**：RMSE 随「云量 / 对流强度 / 场景变化强度」如何变化。

为什么做这个
------------
总体 RMSE 会把不同难度的样本混在一起。分桶后可以看出模型在**哪类场景**上
失效，这是比单一数字更有诊断价值的信息（cb 的 fig2(b) 做了"误差随亮温"，
这里进一步做云量与变化强度）。

⚠️ 关于"时段"分桶
----------------
**做不了**。数据集（test zarr）里**没有时间戳**：
- 上游论文明确说明 patch 是随机抽取的，且 zarr 未保存原始文件名/时间
- cb 的 `current_state.md` 也记录了 "[UNKNOWN] 每个 patch 的原始 GOES 文件时间戳
  不在 Zarr metadata 中"
因此本脚本用**场景变化强度**替代"时段"作为"难度"维度（该量与云的演变快慢直接相关，
且完全可从数据算出）。

三个分桶维度
------------
1. **云量**：样本内 `Tb < 273 K` 的像素占比（273 K ≈ 0 °C，红外云检测的常用阈值）
2. **对流强度**：样本最冷 5% 像素的平均亮温（越低 = 云顶越高/对流越强）
3. **场景变化强度**：`mean|GT − 上一帧|`（K），即 10 分钟内云系变化幅度

产出
----
* `fig_error_buckets.png` —— 三面板，每面板为各分位桶的 SimVP / persistence RMSE
* `error_buckets.json`    —— 全部数值

用法::
    python tools/error_buckets.py
"""

import argparse
import json
import math
import os
import os.path as osp
import sys

_HERE = osp.dirname(osp.abspath(__file__))
_OPENSTL_DIR = osp.dirname(_HERE)
if _OPENSTL_DIR not in sys.path:
    sys.path.insert(0, _OPENSTL_DIR)

import numpy as np  # noqa: E402
from openstl.datasets.dataloader_satcast import MEAN_K, STD_K  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
# 中文字体：matplotlib 默认 DejaVu Sans 无 CJK 字形，不设置会让图上中文变方框
from plot_style import use_cjk_font  # noqa: E402

use_cjk_font()


RUN_DIR = osp.join(_OPENSTL_DIR, 'work_dirs', 'satcast_simvp_full')
OUT_DIR = osp.join(_OPENSTL_DIR, 'docs', 'experiments',
                   'EXP-004-simvp-vs-baseline')
CLOUD_TB_THRESH = 273.0        # K，红外云检测常用阈值
NBINS = 5


def to_K(x):
    return np.asarray(x, dtype=np.float64) * STD_K + MEAN_K


def rmse_k(pred, true):
    """逐样本 RMSE (K)。pred/true: (N, H, W) 归一化空间。"""
    return np.sqrt(((pred - true) ** 2).mean(axis=(1, 2))) * STD_K


def bucket_stats(values, metric_s, metric_p, n_bins=NBINS, label=''):
    """按 values 分位分桶，返回每桶的 SimVP / persistence RMSE 均值与标准误。"""
    qs = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, qs)
    idx = np.clip(np.digitize(values, edges[1:-1]), 0, n_bins - 1)
    out = []
    for k in range(n_bins):
        m = idx == k
        if m.sum() < 5:
            out.append(None)
            continue
        out.append({
            'bin': k, 'n': int(m.sum()),
            'var_median': float(np.median(values[m])),
            'var_range': [float(values[m].min()), float(values[m].max())],
            'simvp_rmse_K': float(metric_s[m].mean()),
            'simvp_sem': float(metric_s[m].std(ddof=1) / math.sqrt(m.sum())),
            'persist_rmse_K': float(metric_p[m].mean()),
            'persist_sem': float(metric_p[m].std(ddof=1) / math.sqrt(m.sum())),
        })
    print(f'  [{label}] 分桶完成，各桶样本数 = '
          f'{[ (b["n"] if b else 0) for b in out ]}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default=RUN_DIR)
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sub = osp.join(args.run_dir, 'baseline_eval')
    pred = np.load(osp.join(sub, 'preds.npy'), mmap_mode='r')[0, :, 0].astype(np.float64)
    true = np.load(osp.join(sub, 'trues.npy'), mmap_mode='r')[0, :, 0].astype(np.float64)
    pers = np.load(osp.join(sub, 'persist.npy'), mmap_mode='r')[0, :, 0].astype(np.float64)
    n = pred.shape[0]
    print(f'[INFO] 单步数据 {pred.shape}（{n} 条）')

    # 逐样本指标
    rs = rmse_k(pred, true)
    rp = rmse_k(pers, true)

    # 真值亮温 + 各分桶变量
    tb = to_K(true)
    cloud_frac = (tb < CLOUD_TB_THRESH).mean(axis=(1, 2)) * 100.0       # %
    cold_tail = np.percentile(tb.reshape(n, -1), 5, axis=1)              # 最冷5%的Tb
    change = np.abs(true - pers).mean(axis=(1, 2)) * STD_K               # K

    print(f'[INFO] 亮温分布 (K): p1={np.percentile(tb,1):.1f} '
          f'p25={np.percentile(tb,25):.1f} p50={np.percentile(tb,50):.1f} '
          f'p75={np.percentile(tb,75):.1f} p99={np.percentile(tb,99):.1f}')
    print(f'[INFO] 云量(Tb<{CLOUD_TB_THRESH:.0f}K)占比: '
          f'中位 {np.median(cloud_frac):.1f}%  范围 '
          f'[{cloud_frac.min():.1f}, {cloud_frac.max():.1f}]%')

    buckets = {
        'cloud_fraction_pct': bucket_stats(cloud_frac, rs, rp, label='云量'),
        'coldest5pct_Tb_K': bucket_stats(cold_tail, rs, rp, label='对流强度'),
        'scene_change_K': bucket_stats(change, rs, rp, label='场景变化'),
    }

    # ------------------------------ 画图 ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    titles = [
        ('(a) 按云量分桶（Tb < 273 K 的像素占比）', '云量 (%)'),
        ('(b) 按对流强度分桶（最冷 5% 像素的亮温）', '最冷 5% 像素 Tb (K)'),
        ('(c) 按场景变化强度分桶（10 min 内变化幅度）', 'mean|GT − 上一帧| (K)'),
    ]
    for ax, (key, (title, xlabel)) in zip(axes, zip(buckets, titles)):
        bs = [b for b in buckets[key] if b]
        x = np.arange(len(bs))
        xs = [b['var_median'] for b in bs]
        ys = [b['simvp_rmse_K'] for b in bs]
        es = [b['simvp_sem'] for b in bs]
        yp = [b['persist_rmse_K'] for b in bs]
        ep = [b['persist_sem'] for b in bs]
        ax.errorbar(x, ys, yerr=es, fmt='o-', color='#1f77b4',
                    capsize=3, label='SimVP')
        ax.errorbar(x, yp, yerr=ep, fmt='s--', color='#7f7f7f',
                    capsize=3, label='persistence')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:.1f}' for v in xs])
        ax.set_xlabel(xlabel + '（桶中位）')
        ax.set_ylabel('RMSE (K)')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle('误差的分组诊断（单步，test 全量 1024 条；误差棒为标准误）',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = osp.join(args.out_dir, 'fig_error_buckets.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f'[SAVE] 分桶诊断 -> {p}')

    with open(osp.join(args.out_dir, 'error_buckets.json'), 'w') as f:
        json.dump({'n_samples': int(n),
                   'cloud_Tb_threshold_K': CLOUD_TB_THRESH,
                   'overall': {'simvp_rmse_K': float(rs.mean()),
                               'persist_rmse_K': float(rp.mean())},
                   'buckets': buckets}, f, indent=2)
    print('[SAVE] 数值 -> ' + osp.join(args.out_dir, 'error_buckets.json'))


if __name__ == '__main__':
    main()
