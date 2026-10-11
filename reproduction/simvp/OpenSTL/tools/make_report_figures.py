#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""复刻 cb 报告中的分析图（fig1 训练曲线 + fig2 误差分析），用于本项目的 SimVP。

对齐 cb 的口径
-------------
* **fig1 训练曲线**：纵轴换算为**亮温 K 空间的 RMSE**，与主结果同量纲。
  换算：``RMSE(K) = sqrt(MSE) * STD_K``（训练日志里的 loss 是归一化空间的 MSE）。
  画出 train / val 曲线，并用虚线标出 test 集实测 RMSE；标注 SLURM 重启处。
* **fig2 误差分析**（cb 原图四个子图）：
  (a) 逐样本 RMSE 分布直方图（SimVP vs persistence）
  (b) 误差随目标亮温的变化（分箱取中位数）
  (c)(d) 空间平均有符号偏差图（共用对称色标 ±0.4 K；蓝=预测偏冷，红=预测偏暖）

数据用**单步**（LT=1，test 全量 1024 条）——与 cb 的误差分析口径一致。

用法::
    python tools/make_report_figures.py
"""

import glob
import json
import math
import os
import os.path as osp
import re
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

RUN_DIR = osp.join(_OPENSTL_DIR, 'work_dirs', 'satcast_simvp_full')
OUT_DIR = osp.join(_OPENSTL_DIR, 'docs', 'experiments',
                   'EXP-004-simvp-vs-baseline')
LOG_RE = re.compile(
    r'Epoch\s+(\d+):\s+Lr:\s+([\d.eE+-]+)\s+\|\s+Train Loss:\s+([\d.eE+-]+)'
    r'\s+\|\s+Vali Loss:\s+([\d.eE+-]+)')


def mse_to_rmse_k(mse):
    return math.sqrt(max(mse, 0.0)) * STD_K


def parse_logs(run_dir):
    """解析训练日志，返回 (epochs, train_rmse_K, val_rmse_K, restart_epochs)。"""
    files = sorted(glob.glob(osp.join(run_dir, 'train_*.log')))
    if not files:
        print('[WARN] 未找到训练日志')
        return [], [], [], []
    rec = {}
    restarts = []
    prev_file_last_epoch = None
    for f in files:
        last_ep = None
        with open(f) as fh:
            for line in fh:
                m = LOG_RE.search(line)
                if not m:
                    continue
                ep = int(m.group(1))
                tl, vl = float(m.group(3)), float(m.group(4))
                rec[ep] = (mse_to_rmse_k(tl), mse_to_rmse_k(vl))
                last_ep = ep
        # 两个日志文件之间 = 一次 SLURM 重启（续跑）
        if prev_file_last_epoch is not None and last_ep is not None:
            restarts.append((prev_file_last_epoch, last_ep))
        prev_file_last_epoch = last_ep
        print(f'[INFO] 解析 {osp.basename(f)}：{len(rec)} 条累计，末 epoch={last_ep}')
    eps = sorted(rec)
    return eps, [rec[e][0] for e in eps], [rec[e][1] for e in eps], restarts


def fig_training(eps, tr, va, restarts, test_rmse):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(eps, tr, '-', color='#1f77b4', lw=2, label='train RMSE (K)')
    ax.plot(eps, va, '-', color='#ff7f0e', lw=2, label='validation RMSE (K)')
    if test_rmse:
        ax.axhline(test_rmse, color='k', ls='--', lw=1.6,
                   label=f'test RMSE = {test_rmse:.3f} K')
    for a, b in restarts:
        ax.axvspan(a, b, color='#ffbb78', alpha=0.35,
                   label='SLURM 重启（续跑）' if a == restarts[0][0] else None)
    ax.set_xlabel('epoch')
    ax.set_ylabel('RMSE (brightness temperature, K)')
    ax.set_title('SimVP 训练曲线（纵轴已换算为亮温 K，与主结果同量纲）')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = osp.join(OUT_DIR, 'fig_training_curve.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f'[SAVE] 训练曲线 -> {p}')
    if eps:
        print(f'       train {tr[0]:.3f} -> {tr[-1]:.3f} K | '
              f'val {va[0]:.3f} -> {va[-1]:.3f} K')


def fig_error_analysis(pred, true, persist):
    """pred/true/persist: (N, H, W) 归一化空间。"""
    # (a) 逐样本 RMSE(K)
    rs = np.sqrt(((pred - true) ** 2).mean(axis=(1, 2))) * STD_K
    rp = np.sqrt(((persist - true) ** 2).mean(axis=(1, 2))) * STD_K

    # (b) 目标亮温（用真值的样本均值代表该样本的亮温水平）
    tb = np.asarray(true, dtype=np.float64).mean(axis=(1, 2)) * STD_K + MEAN_K
    qs = np.linspace(0, 100, 11)
    edges = np.percentile(tb, qs)
    bins = np.clip(np.digitize(tb, edges[1:-1]), 0, 8)
    med_s = [np.median(rs[bins == k]) if np.any(bins == k) else np.nan
             for k in range(9)]
    med_p = [np.median(rp[bins == k]) if np.any(bins == k) else np.nan
             for k in range(9)]
    centers = [np.median(tb[bins == k]) if np.any(bins == k) else np.nan
               for k in range(9)]

    # (c)(d) 空间平均有符号偏差 (K)
    bias_s = (pred - true).mean(axis=0) * STD_K
    bias_p = (persist - true).mean(axis=0) * STD_K

    fig, axes = plt.subplots(1, 4, figsize=(23, 5.2))

    axes[0].hist(rs, bins=40, alpha=0.65, color='#1f77b4',
                 label=f'SimVP (mean {rs.mean():.2f} K)')
    axes[0].hist(rp, bins=40, alpha=0.65, color='#7f7f7f',
                 label=f'persistence (mean {rp.mean():.2f} K)')
    axes[0].set_xlabel('per-sample RMSE (K)')
    axes[0].set_ylabel('样本数')
    axes[0].set_title('(a) 逐样本 RMSE 分布')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(centers, med_s, 'o-', color='#1f77b4', label='SimVP')
    axes[1].plot(centers, med_p, 's--', color='#7f7f7f', label='persistence')
    axes[1].set_xlabel('目标亮温 (K)')
    axes[1].set_ylabel('该区间 RMSE 中位数 (K)')
    axes[1].set_title('(b) 误差随目标亮温的变化')
    axes[1].invert_xaxis()          # 亮温越低（云越高/越冷）放左边
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    vmax = 0.4
    for ax, bmp, name in ((axes[2], bias_s, 'SimVP'), (axes[3], bias_p, 'persistence')):
        im = ax.imshow(bmp, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'空间平均有符号偏差 — {name}\n'
                     f'全局 {bmp.mean():+.3f} K（蓝=偏冷 红=偏暖）', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, label='K')

    fig.suptitle('测试集误差分析（单步，test 全量 1024 条）', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = osp.join(OUT_DIR, 'fig_error_analysis.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f'[SAVE] 误差分析 -> {p}')
    print(f'       逐样本 RMSE: SimVP 均值 {rs.mean():.3f} K / 中位 {np.median(rs):.3f} '
          f'/ 最差 {rs.max():.3f} / 最好 {rs.min():.3f}')
    print(f'       persistence 均值 {rp.mean():.3f} K')
    print(f'       全局偏差: SimVP {bias_s.mean():+.3f} K | persistence {bias_p.mean():+.3f} K')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- fig1 训练曲线 ----
    eps, tr, va, restarts = parse_logs(RUN_DIR)
    test_rmse = None
    mj = osp.join(RUN_DIR, 'baseline_eval', 'baseline_metrics.json')
    if osp.exists(mj):
        with open(mj) as f:
            test_rmse = json.load(f)['per_lead_time']['1']['simvp_K']['rmse_K']
    fig_training(eps, tr, va, restarts, test_rmse)

    # ---- fig2 误差分析（单步，1024 条）----
    p = np.load(osp.join(RUN_DIR, 'baseline_eval', 'preds.npy'), mmap_mode='r')
    t = np.load(osp.join(RUN_DIR, 'baseline_eval', 'trues.npy'), mmap_mode='r')
    q = np.load(osp.join(RUN_DIR, 'baseline_eval', 'persist.npy'), mmap_mode='r')
    # 形状 (1, N, C, H, W) -> (N, H, W)
    pred = np.asarray(p[:, :, 0], dtype=np.float64)[0]
    true = np.asarray(t[:, :, 0], dtype=np.float64)[0]
    pers = np.asarray(q[:, :, 0], dtype=np.float64)[0]
    print(f'[INFO] 单步数据 preds={p.shape} -> {pred.shape}')
    fig_error_analysis(pred, true, pers)


if __name__ == '__main__':
    main()
