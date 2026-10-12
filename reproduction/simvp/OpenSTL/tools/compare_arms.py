#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**抗模糊方案对照**：baseline / GAN / 多步MSE / 多步+scheduled sampling 四臂统一对比。

背景
----
长时效（尤其 180 min）预测"越推越模糊"，baseline 在 LT=18 的锐度比只有 0.273
（预测梯度幅值仅为真实图的 27%）。为改善这一点做了两组方案：

* **GAN**：训练时滚动 K 步，用判别器逼迫输出更"真实"
  → 实测：**所有时效 RMSE 都变差**（LT=18 +34.8%~85.8%），且锐度比过冲到 ~3.0（噪声）
  → 与感知-失真权衡一致：GAN 可改善观感，但不可能降低 MSE
* **多步训练 ± scheduled sampling**：不引入生成式损失，对全部 K 步算 MSE，
  并按概率 p 喂模型自己的预测以减少曝光偏差
  → 本脚本用于判断它是否能在**不牺牲 RMSE** 的前提下改善长时效

对比口径
--------
* 数据：test 全量 **1024 条**，滚动 **18 步**（LT=1..18）
* 指标：逐样本 RMSE(K)（配对到同一批样本）+ **锐度比**（预测梯度幅值 / 真实梯度幅值）
  - 锐度比 < 1 = 偏模糊；≈1 = 与真实相当；>1 = 有高频噪声

⚠️ 对照局限（必须注明）
----------------------
* GAN / 多步臂**训练数据来源不同**：GAN 用 train 集（35595 条，仅 1 帧真值）；
  多步臂用 validation 集（800 条，18 帧真值）。故为"同起点、同步数预算的方向性对照"，
  非严格同数据对照。
* 各臂的热启动起点相同（baseline best.ckpt），梯度步数预算相同（8000 步）。

用法::
    python tools/compare_arms.py
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
import torch  # noqa: E402
from openstl.datasets.dataloader_satcast import STD_K  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# 四臂：(显示名, 预测子目录, 颜色, 说明)
ARMS = [
    ('baseline', 'satcast_simvp_full', '#1f77b4',
     '单步 MSE（teacher forcing），100 epoch'),
    ('GAN (K=6)', 'satcast_simvp_gan', '#d62728',
     '多步滚动 + 判别器对抗损失（相对权重 5%）+ 双向锐度正则'),
    ('multi-step (p=0)', 'satcast_simvp_rollout_ss0.0', '#2ca02c',
     '多步 MSE，滚 6 步，teacher forcing'),
    ('multi-step + SS (p=0.5)', 'satcast_simvp_rollout_ss0.5', '#9467bd',
     '多步 MSE + scheduled sampling 概率 0.5'),
]
OUT_DIR = osp.join(_OPENSTL_DIR, 'docs', 'experiments', 'EXP-006-anti-blur')
SUB = 'baseline_eval_rollout18_full'


def grad_magnitude(x):
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    return ((dy.mean() + dx.mean()) / 2).item()


def load_arm(d):
    f = osp.join(_OPENSTL_DIR, 'work_dirs', d, SUB, 'preds.npy')
    if not osp.exists(f):
        return None
    return np.load(f, mmap_mode='r')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 真值以 baseline 目录为准（各臂评估的是同一份 test 数据）
    gt = np.load(osp.join(_OPENSTL_DIR, 'work_dirs', ARMS[0][1], SUB,
                          'trues.npy'), mmap_mode='r')

    results, avail = {}, []
    for name, d, color, desc in ARMS:
        p = load_arm(d)
        if p is None:
            print(f'[WARN] 缺少 {d}/{SUB}/preds.npy，跳过该臂')
            continue
        n = min(p.shape[1], gt.shape[1])
        rmse, sharp = [], []
        for j in range(p.shape[0]):
            # 注意形状：preds 为 (LT, N, C, H, W)，故 p[j, :n] 得到 (n, C, H, W)。
            # 若写成 p[j, :n, 0] 会去掉通道维变成 3D，grad_magnitude 会报
            # "too many indices for tensor of dimension 3"。
            pr = torch.from_numpy(np.array(p[j, :n], dtype=np.float32))
            tr = torch.from_numpy(np.array(gt[j, :n], dtype=np.float32))
            rmse.append(math.sqrt(float(((pr - tr) ** 2).mean())) * STD_K)
            sharp.append(grad_magnitude(pr) / (grad_magnitude(tr) + 1e-12))
        results[name] = {'rmse': rmse, 'sharp': sharp, 'color': color,
                         'desc': desc}
        avail.append(name)
        print(f'[OK] {name:26s} LT1 RMSE={rmse[0]:7.3f}K  '
              f'LT18 RMSE={rmse[-1]:7.3f}K  锐度LT18={sharp[-1]:.3f}')

    if len(avail) < 2:
        print('[FATAL] 可用臂不足 2 个，无法对比')
        return

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    lts = list(range(1, 19))
    for name in avail:
        r = results[name]
        axes[0].plot(lts, r['rmse'], '-', color=r['color'], lw=2, label=name)
        axes[1].plot(lts, r['sharp'], '-', color=r['color'], lw=2, label=name)
    axes[1].axhline(1.0, color='k', ls='--', lw=1.2, label='锐度=1（与真实相当）')
    for ax, ylabel, title in (
            (axes[0], 'RMSE (K)', '(a) RMSE 随 lead time（越低越好）'),
            (axes[1], '锐度比 pred/real', '(b) 锐度比（<1 模糊，>1 噪声）')):
        ax.set_xlabel('lead time（步 × 10 min）')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('抗模糊方案对照（test 全量 1024 条 × 18 步）', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pfig = osp.join(args.out_dir, 'fig_anti_blur_compare.png')
    fig.savefig(pfig, dpi=130)
    plt.close(fig)
    print(f'[SAVE] 对比图 -> {pfig}')

    # ---------------- 表 + 文档 ----------------
    base = results.get('baseline')
    L = []
    L.append('## 四臂对照结果（自动生成）')
    L.append('')
    L.append('> 由 `tools/compare_arms.py` 生成。数据：test 全量 1024 条，滚动 18 步。')
    L.append('')
    L.append('### 逐时效 RMSE（K）')
    L.append('')
    L.append('| LT | ' + ' | '.join(avail) + ' |')
    L.append('|---|' + '---|' * len(avail))
    for j in lts:
        row = [f'{j:2d}']
        for name in avail:
            v = results[name]['rmse'][j - 1]
            mark = ' **' if name != 'baseline' and base and v < base['rmse'][j - 1] else ' '
            row.append(f'{mark}{v:.3f}{mark.strip()}')
        L.append('| ' + ' | '.join(row) + ' |')
    L.append('')
    L.append('### 锐度比（pred 梯度 / real 梯度）')
    L.append('')
    L.append('| LT | ' + ' | '.join(avail) + ' |')
    L.append('|---|' + '---|' * len(avail))
    for j in lts:
        L.append('| %2d | ' % j + ' | '.join(f"{results[n]['sharp'][j-1]:.3f}"
                                             for n in avail) + ' |')

    L.append('')
    L.append('### 结论（自动判定）')
    L.append('')
    if base:
        b18 = base['rmse'][-1]
        for name in avail:
            if name == 'baseline':
                continue
            r18 = results[name]['rmse'][-1]
            s18 = results[name]['sharp'][-1]
            d = (r18 / b18 - 1) * 100
            if d < -2:
                verdict = f'**RMSE 改善 {abs(d):.1f}%** ✅'
            elif d > 2:
                verdict = f'RMSE 恶化 {d:.1f}% ❌'
            else:
                verdict = 'RMSE 基本持平 —'
            L.append(f'- **{name}**：LT=18 {verdict}；锐度比 {s18:.3f}'
                     f'（baseline {base["sharp"][-1]:.3f}）'
                     f'{"（仍偏模糊）" if s18 < 0.9 else ("（接近真实）" if s18 < 1.15 else "（过冲/噪声）")}')

    L.append('')
    L.append('> ⚠️ 对照局限：GAN 臂用 train 集（35595 条，1 帧真值）训练；多步臂用 '
             'validation 集（800 条，18 帧真值）训练。两者**训练数据来源不同**，'
             '本对照比较的是"同热启动起点、同 8000 步预算下哪种改进方向更有效"。')

    body = '\n'.join(L)
    readme = osp.join(args.out_dir, 'README.md')
    header = (
        '# EXP-006 — 抗模糊方案对照（GAN vs 多步训练 / scheduled sampling）\n'
        '\n'
        '## Identity\n'
        '- 实验 ID：`EXP-006`\n'
        '- 状态：`AUTO-GENERATED`\n'
        '- 日期：{}\n'
        '- 关联：`EXP-004`（baseline 与统一协议）、`EXP-005`（多 seed 对照）\n'
        '\n'
        '## 目的\n'
        '长时效预测"越推越模糊"（baseline 在 LT=18 锐度比仅 0.273）。\n'
        '本实验对照两类改进方向：\n'
        '1. **GAN**：判别器逼迫输出更真实（生成式，会引入纹理）\n'
        '2. **多步训练 ± scheduled sampling**：纯监督，减少曝光偏差（不引入纹理）\n'
        '\n'
        '## 设置\n'
        '- 各臂**热启动**于同一个 baseline `best.ckpt`，梯度步数预算均为 **8000 步**\n'
        '- 评估：test 全量 1024 条，滚动 18 步，逐样本 RMSE(K) 与锐度比\n'
        '\n'
    ).format(__import__('datetime').datetime.now().strftime('%Y-%m-%d'))

    if osp.exists(readme):
        text = open(readme).read()
    else:
        text = header + '<!-- ARMS:BEGIN -->\n<!-- ARMS:END -->\n'
    if '<!-- ARMS:BEGIN' not in text or '<!-- ARMS:END -->' not in text:
        print('[WARN] README 无 ARMS 标记，跳过写入')
    else:
        b = text.index('<!-- ARMS:BEGIN')
        b_end = text.index('-->', b) + 3
        e = text.index('<!-- ARMS:END')
        e_end = text.index('-->', e) + 3
        text = (text[:b] + '<!-- ARMS:BEGIN -->\n' + body + '\n'
                + '<!-- ARMS:END -->' + text[e_end:])
        with open(readme, 'w') as f:
            f.write(text)
        print(f'[SAVE] 文档 -> {readme}')

    with open(osp.join(args.out_dir, 'arms_metrics.json'), 'w') as f:
        json.dump({n: {'rmse': results[n]['rmse'], 'sharp': results[n]['sharp'],
                       'desc': results[n]['desc']} for n in avail}, f, indent=2)

    print('=' * 70)
    print(body)
    print('=' * 70)


if __name__ == '__main__':
    main()
