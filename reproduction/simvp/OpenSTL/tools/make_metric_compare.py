#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**跨臂 8 指标横向对比图**：baseline / GAN / GAN+SS 与真值并排。

为什么需要这个脚本
------------------
`docs/experiments/EXP-008-metrics-baseline/<arm>/` 下已有**逐臂**的
bt_pdf.png / psd_radial.png / coverage_trend.png，但三臂各画一套、
图与图之间要人脑比对。本脚本把三臂画到**同一张图**上，直接回答：

* 哪个尺度先塌（PSD 三臂叠加）
* 谁的 BT 分布过早向均值收缩（PDF 三臂叠加）
* 逐时效的 SSIM / CSI / FSS / 锐度占比谁更好
* 位移误差、冷却率误差、云量偏差谁更大
* 池化总表（可直接与 model_performance / dcm 的表横向比）

数据源（三臂同结构、同口径，已核对）
----------------------------------
`docs/experiments/EXP-008-metrics-baseline/{unet,baseline,gan,persistence}/metrics_8.json`
字段：`meta.pdf_bins_K`、`bt_pdf_density{pred,truth}`、`psd_radial{k,pred,truth}`、
`per_lead[str(i)]{rmse_K,mae_K,ssim,csi,fss,coverage_*,hf_energy_frac_*,
lapvar_*,lapvar_frac_of_truth}`、`pooled{...}`、`trend[str(k)]{...}`。

⚠️ 引用注意
-----------
* SSIM/CSI/FSS 口径 = `zjt flowcast_v2/datasets/goes/metrics.py`
  （gaussian σ=1.5、data_range=180.008、FSS 窗 9、CSI 阈值 Tb≤235K）——
  与组内已发表表格可比，但**不要**与 `eval_simvp_baseline.py` 的 SSIM 混排
  （那是 11×11 均值池化 + 固定常数，另一套实现）。
* CRPS 对单成员确定性模型解析退化为 MAE（脚本不画 CRPS 曲线，只在小结里注明）。
* `trend` 里的冷却率/面积增长用**真值**界定冷芯区域，属诊断型指标，
  不能当作业务评分。

用法::

    python tools/make_metric_compare.py
    python tools/make_metric_compare.py
    # 或只画子集：--arms unet baseline gan
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

EXP8 = osp.join(_OPENSTL_DIR, 'docs', 'experiments', 'EXP-008-metrics-baseline')

# 臂的显示名、目录名、颜色、线型。
# ⚠️ 用户明确要求：**不出现 scheduled sampling（SS）相关内容、不与 SS/GAN+SS 对比**；
#    **统一以 UNet（official）作为 baseline/参照基准**，故 UNet 置首行。
# 因此本图只画 4 臂：UNet（基准）/ SimVP（ours）/ GAN（K=6）/ persistence。
# `gan_ss` 目录的数据虽仍在盘上（历史实验证据），但**一律不画、不参与对比**。
# 主对比臂（画在逐时效曲线 / PSD / PDF 上，6 条线以防拥挤）。
# ⚠️ 用户明确要求：**不出现 scheduled sampling（SS）相关内容、不与 SS/GAN+SS 对比**；
#    **统一以 UNet（official）作为 baseline/参照基准**，故 UNet 置首行。
# `gan_ss` 目录的数据虽仍在盘上（历史实验证据），但**一律不画、不参与对比**。
ARMS = [
    ('UNet（official，基准）', 'unet', '#d62728', 's-'),
    ('SimVP（ours）', 'baseline', '#1f77b4', 'o-'),
    ('GAN（K=6）', 'gan', '#ff7f0e', 'D-'),
    ('纯残差 R0（无 GAN）', 'residual_R0_nogan', '#9467bd', 'p-'),
    ('残差+GAN R0', 'resgan_R0', '#2ca02c', '^-'),
    ('平流（冻结流 α=.75）', 'adv_frozen_a075', '#8c564b', 'v-'),
    ('平流（自一致 α=1）', 'adv_sc_a1', '#e377c2', '*-'),
    ('persistence', 'persistence', '#7f7f7f', 'x--'),
]
# 残差分支消融臂（单独一张图，回答"平流基有没有用 / GAN 有没有用"）
ABLATION = [
    ('SimVP（ours）', 'baseline', '#1f77b4', 'o-'),
    ('纯残差 R0（无 GAN）', 'residual_R0_nogan', '#9467bd', 'p-'),
    ('残差+GAN R0（无平流基）', 'resgan_R0', '#2ca02c', '^-'),
    ('残差+GAN R1（含平流基）', 'resgan_R1', '#17becf', 'P-'),
    ('残差+GAN R1sham（基填零）', 'resgan_R1sham', '#8c564b', 'X-'),
    ('残差+GAN R0（train35083,K=1）', 'resgan_R0_bigdata', '#e377c2', '*-'),
]
TRUTH_C = '#333333'


def load(arm_dir):
    p = osp.join(EXP8, arm_dir, 'metrics_8.json')
    if not osp.exists(p):
        return None
    return json.load(open(p, encoding='utf-8'))


def first(data):
    """返回第一个**有数据**的臂，用于读公共元信息（n_lead / pdf_bins 等）。

    不能用 `data[ARMS[0][1]]` 硬取 baseline：用户可能用 `--arms gan` 只画一臂，
    那时 baseline 为 None，直接取会 TypeError。
    """
    for _, d, _, _ in ARMS:
        if data.get(d):
            return data[d]
    raise KeyError('没有任何臂的数据')


def _cjk():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_style import use_cjk_font
    use_cjk_font()
    return plt


# ------------------------------------------------------------------ 分图 ---
def fig_psd(data, out):
    """径向平均功率谱：三臂叠加 + 真值，回答"哪个尺度先塌"。"""
    plt = _cjk()
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    t = first(data)
    k = np.array(t['psd_radial']['k'])
    ax.loglog(k[1:], np.array(t['psd_radial']['truth'])[1:], '-',
              color=TRUTH_C, lw=2.4, label='真值 truth')
    for disp, d, c, st in ARMS:
        m = data[d]
        if not m:
            continue
        ax.loglog(k[1:], np.array(m['psd_radial']['pred'])[1:], st,
                  color=c, lw=1.9, ms=4, label=disp)
    ax.axvline(40, color='#999999', ls=':', lw=1.2, label='高频阈值 |k|>40')
    ax.set_xlabel('径向波数 |k|（像素⁻¹）', fontsize=11)
    ax.set_ylabel('功率谱密度', fontsize=11)
    ax.set_title('径向平均功率谱：哪个空间尺度先塌（test 全量 1024 条）',
                 fontsize=12)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=10)
    p = osp.join(out, 'compare_psd.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f'  [SAVE] {p}')


def fig_pdf(data, out):
    """BT 概率密度：三臂叠加 + 真值，看谁过早向均值收缩。"""
    plt = _cjk()
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    t = first(data)
    bins = np.array(t['meta']['pdf_bins_K'])
    c = 0.5 * (bins[:-1] + bins[1:])
    ax.plot(c, t['bt_pdf_density']['truth'], '-', color=TRUTH_C, lw=2.4,
            label='真值 truth')
    for disp, d, col, _ in ARMS:
        m = data[d]
        if not m:
            continue
        ax.plot(c, m['bt_pdf_density']['pred'], '--', color=col, lw=2.0,
                label=disp)
    ax.set_xlabel('亮度温度 (K)', fontsize=11)
    ax.set_ylabel('概率密度', fontsize=11)
    ax.set_title('BT 概率密度函数（全时效合计）：越"尖"越像真值，'
                 '越"平"越向均值收缩', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    p = osp.join(out, 'compare_pdf.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f'  [SAVE] {p}')


def fig_perlead(data, out):
    """逐时效 2×2：RMSE / SSIM / CSI / FSS。"""
    plt = _cjk()
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4), constrained_layout=True)
    n_lead = int(first(data)['meta']['n_lead'])
    lts = np.arange(1, n_lead + 1)
    panels = [
        ('rmse_K', 'RMSE（亮温 K）—— 越低越好', True),
        ('ssim', 'SSIM（组内口径）—— 越高越好', False),
        ('csi', f'CSI（Tb≤{first(data)["meta"]["cold_thresh_K"]:.0f}K）'
                '—— 越高越好', False),
        ('fss', 'FSS（窗 9）—— 越高越好', False),
    ]
    for ax, (key, title, lower_better) in zip(axes.ravel(), panels):
        for disp, d, col, st in ARMS:
            m = data[d]
            if not m:
                continue
            y = [m['per_lead'][str(i)][key] for i in lts]
            ax.plot(lts, y, st, color=col, lw=1.9, ms=4, label=disp)
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel('预报时效（步，1 步 = 10 分钟）', fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    p = osp.join(out, 'compare_perlead.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f'  [SAVE] {p}')


def fig_sharpness(data, out):
    """逐时效锐度占比（LapVar/真值）+ 高频能量占比：回答"第几步开始糊"。"""
    plt = _cjk()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    n_lead = int(first(data)['meta']['n_lead'])
    lts = np.arange(1, n_lead + 1)

    ax = axes[0]
    for disp, d, col, st in ARMS:
        m = data[d]
        if not m:
            continue
        y = [m['per_lead'][str(i)]['lapvar_frac_of_truth'] * 100 for i in lts]
        ax.plot(lts, y, st, color=col, lw=1.9, ms=4, label=disp)
    ax.axhline(100, color='#999999', ls='--', lw=1.2, label='与真值同等清晰')
    ax.axhline(50, color='#cccccc', ls=':', lw=1.0, label='50%（锐度崩塌参考线）')
    ax.set_title('LapVar 占真值比例（%）—— 越低越糊', fontsize=11.5)
    ax.set_xlabel('预报时效（步）', fontsize=10)
    ax.set_ylabel('%', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    for disp, d, col, st in ARMS:
        m = data[d]
        if not m:
            continue
        y = [m['per_lead'][str(i)]['hf_energy_frac_pred']
             / max(m['per_lead'][str(i)]['hf_energy_frac_truth'], 1e-12)
             for i in lts]
        ax.plot(lts, y, st, color=col, lw=1.9, ms=4, label=disp)
    ax.axhline(1.0, color='#999999', ls='--', lw=1.2, label='与真值同等高频')
    ax.set_title('高频能量占比（预报/真值，|k|>40）—— 越小越糊', fontsize=11.5)
    ax.set_xlabel('预报时效（步）', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    p = osp.join(out, 'compare_sharpness.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f'  [SAVE] {p}')


def fig_trend(data, out):
    """趋势量 1×3：位移误差 / 冷却率误差 / 云量偏差。"""
    plt = _cjk()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    panels = [
        ('cloud_motion_err_px', '云团移动误差（像素/10min）—— 位移准不准', None),
        ('cooling_rate_err_K', '冷芯冷却率误差（K/10min）\n正=低估冷却', 0.0),
        ('coverage_diff', '冷芯覆盖率偏差（预报−真值）\n负=云量报少', 0.0),
    ]
    n_lead = int(first(data)['meta']['n_lead'])
    for ax, (key, title, hline) in zip(axes, panels):
        for disp, d, col, st in ARMS:
            m = data[d]
            if not m:
                continue
            if key == 'coverage_diff':
                # 覆盖率逐时效直接取 per_lead（trend 只覆盖少数抽样时效）
                xs = list(range(1, n_lead + 1))
                y = [m['per_lead'][str(i)]['coverage_diff'] for i in xs]
            else:
                ks = sorted(int(x) for x in m['trend'])
                xs = ks
                y = [m['trend'][str(i)][key] for i in ks]
            ax.plot(xs, y, st, color=col, lw=1.9, ms=4, label=disp)
        if hline is not None:
            ax.axhline(hline, color='#999999', ls='--', lw=1.0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('预报时效（步）', fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    p = osp.join(out, 'compare_trend.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f'  [SAVE] {p}')


def fig_pooled(data, out):
    """池化 18 时效总表（柱状图，6 项指标），可与组内总表横向比。"""
    plt = _cjk()
    items = [
        ('rmse_K', 'RMSE (K)', False),
        ('ssim', 'SSIM', True),
        ('csi', 'CSI', True),
        ('fss', 'FSS', True),
        ('hf_energy_frac_truth', '真值高频占比\n（参照）', True),
        ('lapvar_frac_of_truth', 'LapVar 占真值比', True),
    ]
    # 高频占比改成"预报/真值"才有可比性
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.8), constrained_layout=True)
    for ax, (key, title, higher_better) in zip(axes.ravel(), items):
        names, vals, cols = [], [], []
        for disp, d, col, _ in ARMS:
            m = data[d]
            if not m:
                continue
            if key == 'hf_energy_frac_truth':
                # 改成「预报/真值」比值才有跨臂可比性（绝对占比受数据本身影响）
                v = m['pooled']['hf_energy_frac_pred'] / max(
                    m['pooled']['hf_energy_frac_truth'], 1e-12)
            else:
                v = m['pooled'][key]
            names.append(disp)
            vals.append(v)
            cols.append(col)
        if not vals:                     # 该臂全缺时不要画空坐标
            ax.axis('off')
            continue
        bars = ax.bar(names, vals, color=cols, edgecolor='k', lw=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9.5)
        label = ('高频能量占比（预报/真值）' if key == 'hf_energy_frac_truth'
                 else title)
        ax.set_title(f'{label}（{"越高越好" if higher_better else "越低越好"}）',
                     fontsize=11)
        ax.grid(alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=9.5)
        # 给柱顶/柱底留余量，避免数值标签被裁掉
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.25 if hi > lo else abs(hi) * 0.2 + 0.05
        ax.set_ylim(lo - pad, hi + pad)
    p = osp.join(out, 'compare_pooled.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f'  [SAVE] {p}')


def fig_ablation(data, out):
    """残差分支消融：1×3（RMSE / CSI / 冷芯覆盖率），5 个残差相关臂。

    直接回答两个预注册问题：
      * 平流基有没有用？  -> 比较 R0（无基）与 R1（含基）、R1sham（基填零）
      * GAN 有没有用？    -> 比较「纯残差 R0」与「残差+GAN R0」
    """
    plt = _cjk()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), constrained_layout=True)
    n_lead = int(first(data)['meta']['n_lead'])
    lts = np.arange(1, n_lead + 1)
    panels = [
        ('rmse_K', 'RMSE（亮温 K）—— 越低越好'),
        ('csi', 'CSI（Tb≤235K）—— 越高越好'),
        ('coverage_diff', '冷芯覆盖率（预报/真值）—— 越接近 1 越好'),
    ]
    for ax, (key, title) in zip(axes, panels):
        for disp, d, col, st in ABLATION:
            m = data.get(d)
            if not m:
                continue
            if key == 'coverage_diff':
                y = [m['per_lead'][str(i)]['coverage_pred']
                     / max(m['per_lead'][str(i)]['coverage_truth'], 1e-12)
                     for i in lts]
            else:
                y = [m['per_lead'][str(i)][key] for i in lts]
            ax.plot(lts, y, st, color=col, lw=1.9, ms=4, label=disp)
        if key == 'coverage_diff':
            ax.axhline(1.0, color='#999999', ls='--', lw=1.1,
                       label='与真值覆盖一致')
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel('预报时效（步）', fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)
    p = osp.join(out, 'compare_ablation.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f'  [SAVE] {p}')


def print_table(data):
    """文本总表（同时写入 compare_summary.txt，便于复制到报告）。"""
    hdr = (f'{"指标（18 时效池化）":<26}' +
           ''.join(f'{d[0]:>14}' for d in ARMS if data[d[1]]))
    lines = [hdr, '-' * len(hdr)]
    rows = [
        ('RMSE (K)', 'rmse_K', '{:.3f}'),
        ('MAE / CRPS (K)', 'mae_K', '{:.3f}'),
        ('SSIM', 'ssim', '{:.4f}'),
        ('CSI (Tb≤235K)', 'csi', '{:.4f}'),
        ('FSS (窗9)', 'fss', '{:.4f}'),
        ('高频能量占比 预报/真值', None, '{:.3f}'),
        ('LapVar 占真值比', 'lapvar_frac_of_truth', '{:.3f}'),
        ('冷芯覆盖率 预报/真值', None, '{:.3f}'),
    ]
    for label, key, fmt in rows:
        cells = []
        for _, d, _, _ in ARMS:
            m = data[d]
            if not m:
                continue
            if key is None and '高频' in label:
                v = m['pooled']['hf_energy_frac_pred'] / max(
                    m['pooled']['hf_energy_frac_truth'], 1e-12)
            elif key is None:
                v = (m['pooled']['coverage_pred']
                     / max(m['pooled']['coverage_truth'], 1e-12))
            else:
                v = m['pooled'][key]
            cells.append(fmt.format(v))
        lines.append(f'{label:<26}' + ''.join(f'{c:>14}' for c in cells))
    txt = '\n'.join(lines)
    print('\n' + txt)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='*', default=None,
                    help='要画的臂（目录名），默认 unet baseline gan persistence')
    ap.add_argument('--out-dir', default=EXP8)
    args = ap.parse_args()

    want = set(args.arms) if args.arms else {d for _, d, _, _ in ARMS}
    data = {}
    for _, d, _, _ in ARMS:
        if d in want:
            m = load(d)
            data[d] = m
            if m is None:
                print(f'[WARN] 缺少 {d}/metrics_8.json，跳过该臂')
    if not data:
        raise SystemExit('没有任何臂的数据，退出。请先跑 tools/eval_metrics.py')
    os.makedirs(args.out_dir, exist_ok=True)

    print('生成跨臂 8 指标对比图 ...')
    fig_psd(data, args.out_dir)
    fig_pdf(data, args.out_dir)
    fig_perlead(data, args.out_dir)
    fig_sharpness(data, args.out_dir)
    fig_trend(data, args.out_dir)
    fig_pooled(data, args.out_dir)
    fig_ablation(data, args.out_dir)
    txt = print_table(data)
    with open(osp.join(args.out_dir, 'compare_summary.txt'), 'w') as f:
        f.write(txt + '\n')
    print(f'\n[DONE] 对比图与总表已写入 {args.out_dir}')


if __name__ == '__main__':
    main()
