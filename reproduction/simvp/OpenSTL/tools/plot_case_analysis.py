#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 case 级可预报性分析图（读 `case_analysis_*.json`）。

产出两张图：
  * `case_tiers.png`     难度分档的逐时效 RMSE 曲线 + 云量分档对比
  * `case_exceed.png`    超阈值步数分布 + 云量-难度散点
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

TIER_C = {'easy(易)': '#2ca02c', 'medium(中)': '#ff7f0e', 'hard(难)': '#d62728'}
BIN_C = {'少云': '#aec7e8', '中': '#1f77b4', '多云': '#08306b'}


def _cjk():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_style import use_cjk_font
    use_cjk_font()
    return plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', default=osp.join(
        _OPENSTL_DIR, 'docs', 'experiments', 'EXP-009-case-analysis'))
    ap.add_argument('--tags', default='baseline,A_cont')
    ap.add_argument('--out-dir', default=osp.join(
        _OPENSTL_DIR, 'docs', 'experiments', 'EXP-009-case-analysis'))
    args = ap.parse_args()
    tags = [t for t in args.tags.split(',') if t.strip()]
    D = {t: json.load(open(osp.join(args.in_dir, f'case_analysis_{t}.json')))
         for t in tags if osp.exists(osp.join(args.in_dir, f'case_analysis_{t}.json'))}
    if not D:
        raise SystemExit('找不到 case_analysis_*.json')
    os.makedirs(args.out_dir, exist_ok=True)
    plt = _cjk()

    # ---------------- 图 1：难度分档逐时效 RMSE ----------------
    tag0 = tags[0]
    d0 = D[tag0]
    rm = np.array(d0['rmse_by_case'])              # (N,18)
    lts = np.arange(1, rm.shape[1] + 1)
    tier = np.array(d0['per_case']['tier'])
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), constrained_layout=True)

    ax = axes[0]
    for name, c in TIER_C.items():
        m = tier == list(TIER_C).index(name)
        if m.sum() == 0:
            continue
        y = rm[m].mean(axis=0)
        lo = np.quantile(rm[m], 0.25, axis=0)
        hi = np.quantile(rm[m], 0.75, axis=0)
        ax.plot(lts, y, '-', color=c, lw=2.2, label=f'{name} (n={m.sum()})')
        ax.fill_between(lts, lo, hi, color=c, alpha=0.15)
    th = d0['meta']['thresh_pool_K']
    ax.axhline(th, color='#666666', ls='--', lw=1.3,
               label=f'阈值 {th:.2f} K')
    ax.set_xlabel('预报时效（步，1 步 = 10 分钟）', fontsize=11)
    ax.set_ylabel('单帧 RMSE（亮温 K）', fontsize=11)
    ax.set_title(f'{tag0}：难度分档的逐时效 RMSE（阴影 = 四分位）', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9.5)

    # 右：云量分档
    ax = axes[1]
    cb = np.array(d0['per_case']['cloud_bin'])
    # ⚠️ 用 json 里记录的 bin_index，不能用 enumerate：dict 只含有数据的桶，
    #    插入顺序与桶索引会错配（曾把 bin1「中」误标成「多云」）。
    for name, v in d0['cloud_bin_stats'].items():
        i = v.get('bin_index', list(d0['cloud_bin_stats']).index(name))
        m = cb == i
        if m.sum() == 0:
            continue
        y = rm[m].mean(axis=0)
        c = BIN_C.get(name, f'C{i}')
        ax.plot(lts, y, '-', color=c, lw=2.2,
                label=f'{name}云 (n={m.sum()})')
    ax.axhline(th, color='#666666', ls='--', lw=1.3, label=f'阈值 {th:.2f} K')
    ax.set_xlabel('预报时效（步，1 步 = 10 分钟）', fontsize=11)
    ax.set_ylabel('单帧 RMSE（亮温 K）', fontsize=11)
    ax.set_title(f'{tag0}：按云量分档（真值冷芯占比三分位）', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9.5)
    p = osp.join(args.out_dir, 'case_tiers.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f'  [SAVE] {p}')

    # ---------------- 图 2：超阈步数分布 + 云量-难度散点 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), constrained_layout=True)

    ax = axes[0]
    dist = d0['exceed_step_dist']
    ks = sorted(int(k) for k in dist)
    vs = [dist[str(k)] for k in ks]
    ax.bar([k * 10 for k in ks], vs, width=8, color='#1f77b4',
           edgecolor='k', lw=0.5)
    ax.set_xlabel('首次超过阈值的时刻（分钟）', fontsize=11)
    ax.set_ylabel('case 数', fontsize=11)
    n_never = d0['n_never_exceed']
    ax.set_title(f'{tag0}：超阈值时刻分布（{n_never} 个 case 全程未超，'
                 f'占 {n_never/d0["meta"]["n_cases"]*100:.0f}%）', fontsize=12)
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    cf = np.array(d0['case_features']['cold_frac'])
    gr = np.array(d0['case_features']['grad_input'])
    km = np.array(d0['per_case']['key_mean_rmse'])
    sc = ax.scatter(cf * 100, gr, c=km, cmap='YlOrRd', s=14, alpha=0.75,
                    edgecolor='none')
    cb_ = ax.figure.colorbar(sc, ax=ax)
    cb_.set_label('四关键时效平均 RMSE (K)', fontsize=10)
    ax.set_xlabel('真值冷芯像素占比 (%)', fontsize=11)
    ax.set_ylabel('输入帧梯度幅值（纹理活跃度）', fontsize=11)
    ax.set_title('难度 = f(云量, 纹理活跃度)', fontsize=12)
    ax.grid(alpha=0.3)
    p = osp.join(args.out_dir, 'case_exceed.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f'  [SAVE] {p}')


if __name__ == '__main__':
    main()
