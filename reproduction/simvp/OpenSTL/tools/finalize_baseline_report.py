#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 baseline 评估结果**自动生成**成 EXP-004 文档的 Results 段。

设计目的：训练跑完后无需人工抄数——本脚本读取 ``eval_simvp_baseline.py`` 产出的
JSON，按统一模板渲染 markdown，替换 ``EXP-004 README.md`` 中
``<!-- RESULTS:BEGIN -->`` 与 ``<!-- RESULTS:END -->`` 之间的内容。

所有对照基线常量（UNet / cb SimVP / persistence）都在此显式定义并标注来源，
避免每次手抄出错。数值来源见 EXP-004 的"独立验证记录"。

用法::

    python tools/finalize_baseline_report.py \
        --single work_dirs/satcast_simvp_full/baseline_eval/baseline_metrics.json \
        --rollout work_dirs/satcast_simvp_full/baseline_eval_rollout18/baseline_metrics.json
"""

import argparse
import json
import math
import os
import os.path as osp
import sys
from datetime import datetime

_HERE = osp.dirname(osp.abspath(__file__))
_OPENSTL_DIR = osp.dirname(_HERE)
if _OPENSTL_DIR not in sys.path:
    sys.path.insert(0, _OPENSTL_DIR)

from openstl.datasets.dataloader_satcast import STD_K  # noqa: E402

BEGIN = '<!-- RESULTS:BEGIN'
END = '<!-- RESULTS:END'

# ---------------------------------------------------------------------------
# 对照基线常量（来源：ly 的 EXP-EVAL-001/003、cb 的 test_metrics.json）
# 均为"复制来源"，勿随意改动；改动需同步 EXP-004 的验证记录表。
# ---------------------------------------------------------------------------
UNET_SINGLE = {                      # ly, 单步 1024 条
    'seed42': {'mse': 0.008092, 'ssim': 0.8913},
    'official': {'mse': 0.008272, 'ssim': 0.8939},
}
UNET_MAE_K_SEED42 = 0.896            # ly 报的 MAE(归一化 0.04634) × STD_K
CB_SIMVP = {'mse': 0.006635, 'rmse_K': 1.5745, 'mae_K': 0.7910}   # cb, 100ep
PERSIST_CB = {'rmse_K': 4.5492, 'mae_K': 2.2022}                  # cb, 1024 条
# ly rollout（128 条子集）：LT -> (seed0 MSE, Official MSE)
UNET_ROLLOUT = {
    1: (0.006878, 0.007019), 3: (0.034210, 0.034390), 6: (0.092090, 0.092310),
    9: (0.163760, 0.164910), 12: (0.249610, 0.248190), 15: (0.353420, 0.352090),
    18: (0.419200, 0.490900),
}
UNET_ROLLOUT_LT18_MEAN = 0.488       # 3-seed 均值（seed0/42/123）


def mse_to_rmse_k(mse):
    return math.sqrt(mse) * STD_K


def load(path):
    if not path or not osp.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_single_table(s):
    a = s['per_lead_time']['1']
    sv, ps = a['simvp'], a['persistence']
    svk, psk = a['simvp_K'], a['persistence_K']
    rows = [
        ('**本项目 SimVP**', sv['mse']['mean'], svk['rmse_K'], svk['mae_K'],
         sv['ssim']['mean']),
        ('SimVP（cb, 100 ep）', CB_SIMVP['mse'], CB_SIMVP['rmse_K'],
         CB_SIMVP['mae_K'], None),
        ('Vanilla UNet（ly, seed42）', UNET_SINGLE['seed42']['mse'],
         mse_to_rmse_k(UNET_SINGLE['seed42']['mse']), UNET_MAE_K_SEED42,
         UNET_SINGLE['seed42']['ssim']),
        ('Vanilla UNet（ly, Official）', UNET_SINGLE['official']['mse'],
         mse_to_rmse_k(UNET_SINGLE['official']['mse']), None,
         UNET_SINGLE['official']['ssim']),
        ('persistence', ps['mse']['mean'], psk['rmse_K'], psk['mae_K'], None),
    ]
    out = ['| 模型 | MSE(归一化) | RMSE(K) | MAE(K) | SSIM |',
           '|---|---|---|---|---|']
    for name, mse, rmse, mae, ssim in rows:
        out.append('| {} | {:.6f} | {:.4f} | {} | {} |'.format(
            name, mse, rmse,
            '{:.4f}'.format(mae) if mae is not None else '—',
            '{:.4f}'.format(ssim) if ssim is not None else '—'))
    return out, a


def build_rollout_table(r):
    per = r['per_lead_time']
    lts = sorted(per.keys(), key=int)
    lines = ['| LT | SimVP MSE | RMSE(K) | persistence(K) | UNet seed0 MSE | '
             'UNet Official MSE |', '|---|---|---|---|---|---|']
    cross = None
    for lt in lts:
        e = per[lt]
        sm = e['simvp']['mse']['mean']
        rk = e['simvp_K']['rmse_K']
        pk = e['persistence_K']['rmse_K']
        note = ''
        if cross is None and rk > pk:
            cross = int(lt)
            note = ' ← 起劣于 persistence'
        u = UNET_ROLLOUT.get(int(lt))
        u_txt = '{:.6f} | {:.6f}'.format(u[0], u[1]) if u else '— | —'
        lines.append('| {}{} | {:.6f} | {:.4f} | {:.4f} | {} |'.format(
            lt, note, sm, rk, pk, u_txt))
    return lines, cross, per


def render(single, rollout):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    out = []
    out.append('## Results — 本项目 SimVP（**自动生成**于 {}）'.format(now))

    if single is None:
        out.append('')
        out.append('> ⚠️ 未找到单步评估结果 JSON，无法生成正式结果。')
        return '\n'.join(out)

    a = single['per_lead_time']['1']
    epoch = single.get('ckpt_epoch', 'N/A')
    n = single.get('n_samples', 'N/A')
    params = single.get('params_M', 'N/A')

    out.append('')
    out.append('> 本段由 `tools/finalize_baseline_report.py` 从评估 JSON 自动生成，'
               '**请勿手工编辑**（改数请改脚本里的基线常量或重跑评估）。')
    out.append('')
    out.append('**checkpoint**：`{}`（best epoch = **{}**，{} 样本评估，'
               '{}M 参数）'.format(single.get('ckpt', 'N/A'), epoch, n, params))

    # ---------------- 场景 1 ----------------
    rows, a = build_single_table(single)
    out.append('')
    out.append('### 场景 1：单步 teacher-forced（test 全量 1024 条，权威数字）')
    out.append('')
    out.extend(rows)
    out.append('')
    red = single.get('mse_reduction_vs_persistence_pct_LT1')
    if red is not None:
        out.append('- MSE 相对 persistence 降低 **{:.2f}%**'.format(red))
    out.append('- persistence **{:.4f} K** 与 cb 独立复算（{:.4f} K）偏差 '
               '**{:.4f}%** → 口径闭环'.format(
                   a['persistence_K']['rmse_K'], PERSIST_CB['rmse_K'],
                   abs(a['persistence_K']['rmse_K'] / PERSIST_CB['rmse_K'] - 1) * 100))
    gain_u = (1 - a['simvp']['mse']['mean'] / UNET_SINGLE['seed42']['mse']) * 100
    gap_cb = (a['simvp_K']['rmse_K'] / CB_SIMVP['rmse_K'] - 1) * 100
    out.append('- **单步结论**：{}'.format(
        '优于 UNet（MSE 好 **{:.1f}%**）'.format(gain_u) if gain_u > 0
        else '劣于 UNet（MSE 差 **{:.1f}%**）'.format(-gain_u)))
    out.append('  与 cb 的 100-epoch 版相比 RMSE **{}{:.1f}%**'.format(
        '高 ' if gap_cb > 0 else '低 ', abs(gap_cb)))

    # ---------------- 场景 2 ----------------
    if rollout is None:
        out.append('')
        out.append('> ⚠️ 未找到 18 步 rollout 结果 JSON。')
        return '\n'.join(out)

    lines, cross, per = build_rollout_table(rollout)
    out.append('')
    out.append('### 场景 2：18 步自回归 rollout（128 条子集，对齐 ly 协议）')
    out.append('')
    out.extend(lines)
    out.append('')

    lt18 = per.get('18') or per[max(per.keys(), key=int)]
    m18 = lt18['simvp']['mse']['mean']
    out.append('### 关键发现')
    out.append('')
    u18 = UNET_ROLLOUT[18][0]
    out.append('1. **单步强、长时弱**：LT=1 SimVP MSE {:.6f} vs UNet(seed0) {:.6f}（{}）；'
               '但 LT=18 SimVP {:.6f} vs UNet(seed0) {:.6f} → 差 **{:.1f}%**，'
               '比 UNet 3-seed 均值（{:.3f}）差 **{:.1f}%**。'.format(
                   per['1']['simvp']['mse']['mean'], UNET_ROLLOUT[1][0],
                   '略胜' if per['1']['simvp']['mse']['mean'] < UNET_ROLLOUT[1][0] else '略逊',
                   m18, u18, (m18 / u18 - 1) * 100,
                   UNET_ROLLOUT_LT18_MEAN,
                   (m18 / UNET_ROLLOUT_LT18_MEAN - 1) * 100))
    if cross:
        out.append('2. **约 {} 分钟后"预测不如不动"**：从 **LT={}** 起 SimVP 的 RMSE '
                   '反超 persistence，LT=18 达 {:.2f} K vs {:.2f} K。'.format(
                       cross * 10, cross, lt18['simvp_K']['rmse_K'],
                       lt18['persistence_K']['rmse_K']))
    else:
        out.append('2. SimVP 在全部 18 步内均优于 persistence。')
    out.append('3. **可能原因（待验证，勿当结论）**：容量差（SimVP gSTA 4.7M vs '
               'UNet2DModel 47.6M，约 10 倍）；无任何 rollout-aware 训练；'
               '仅单 seed，未测 SimVP 自身方差（对照：ly 测得 UNet 在 LT=18 的 CV=13.4%）。')
    out.append('')
    out.append('> ⚠️ 引用限制：以上为**单 seed** 结果；与 UNet 做因果性对比前建议补 '
               'SimVP 多 seed（≥3）。')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--single', default=osp.join(
        _OPENSTL_DIR, 'work_dirs', 'satcast_simvp_full', 'baseline_eval',
        'baseline_metrics.json'))
    ap.add_argument('--rollout', default=osp.join(
        _OPENSTL_DIR, 'work_dirs', 'satcast_simvp_full',
        'baseline_eval_rollout18', 'baseline_metrics.json'))
    ap.add_argument('--readme', default=osp.join(
        _OPENSTL_DIR, 'docs', 'experiments', 'EXP-004-simvp-vs-baseline', 'README.md'))
    args = ap.parse_args()

    single = load(args.single)
    rollout = load(args.rollout)
    body = render(single, rollout)

    with open(args.readme) as f:
        text = f.read()
    if BEGIN not in text or END not in text:
        raise SystemExit('README 中未找到 RESULTS 标记，拒绝写入：{}'.format(args.readme))

    b = text.index(BEGIN)
    b_end = text.index('-->', b) + 3
    e = text.index(END)
    e_end = text.index('-->', e) + 3
    new_text = (text[:b] + '<!-- RESULTS:BEGIN -->\n' + body + '\n'
                + '<!-- RESULTS:END -->' + text[e_end:])
    with open(args.readme, 'w') as f:
        f.write(new_text)

    print('[OK] Results 段已更新：{}'.format(args.readme))
    print('=' * 70)
    print(body)
    print('=' * 70)


if __name__ == '__main__':
    main()
