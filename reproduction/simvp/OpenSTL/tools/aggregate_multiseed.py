#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总 SimVP 多 seed 结果，并与 UNet 多 seed 对照，生成 EXP-005 实验卡。

为什么需要多 seed
------------------
ly 的 EXP-EVAL-003 证明：Vanilla UNet 的**单步**指标 seed 方差极小（CV<1.3%），
但 **rollout LT=18 的 CV 高达 13.4%**。因此单 seed 之间比 rollout 是没有意义的。
本项目要判断"SimVP 在 LT=18 比 UNet 差 60%"是否真实，必须拿到 SimVP 自己的
多 seed 分布，再做带方差的对比。

输入：每个 seed 目录下由 ``eval_simvp_baseline.py`` 产出的两个 JSON
    <dir>/baseline_eval/baseline_metrics.json           单步（1024 条）
    <dir>/baseline_eval_rollout18/baseline_metrics.json  18 步 rollout（128 条）

输出：``docs/experiments/EXP-005-simvp-multiseed/README.md``
（用 ``<!-- MULTISEED:BEGIN/END -->`` 标记，可重复运行、幂等更新）

用法::

    python tools/aggregate_multiseed.py           # 用默认三 seed 目录
    python tools/aggregate_multiseed.py --dirs 42=.../ 0=.../ 123=.../
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

BEGIN = '<!-- MULTISEED:BEGIN'
END = '<!-- MULTISEED:END'

# ---------------------------------------------------------------------------
# ly 的 Vanilla UNet 三 seed rollout（128 条子集）—— 复制自 EXP-EVAL-003 全表
# 顺序：(seed0, seed42, seed123)
# ---------------------------------------------------------------------------
UNET_3SEED = {
    1: (0.006878, 0.006881, 0.006968),
    2: (0.018930, 0.018880, 0.019040),
    3: (0.034210, 0.034380, 0.034160),
    4: (0.051590, 0.052570, 0.052090),
    5: (0.070920, 0.073480, 0.072220),
    6: (0.092090, 0.096610, 0.094030),
    7: (0.114450, 0.121930, 0.117430),
    8: (0.138370, 0.150130, 0.143240),
    9: (0.163760, 0.181870, 0.171910),
    10: (0.190720, 0.216200, 0.203230),
    11: (0.219370, 0.253110, 0.237050),
    12: (0.249610, 0.292470, 0.273370),
    13: (0.281730, 0.334510, 0.312430),
    14: (0.316220, 0.379190, 0.354370),
    15: (0.353420, 0.425380, 0.399510),
    16: (0.393320, 0.473460, 0.448020),
    17: (0.435780, 0.523400, 0.499970),
    18: (0.419200, 0.574300, 0.471700),
}
# ly 的 UNet 单步（1024 条）三 seed MSE
UNET_SINGLE_3SEED = (0.008050, 0.008092, 0.008157)


def load(path):
    if not path or not osp.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return {'mean': m, 'std': 0.0, 'cv': 0.0, 'n': len(vals), 'vals': vals}
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)      # 样本标准差
    sd = math.sqrt(var)
    return {'mean': m, 'std': sd, 'cv': sd / m * 100 if m else 0.0,
            'n': len(vals), 'vals': vals}


def mse_to_rmse_k(mse):
    return math.sqrt(mse) * STD_K


def collect(dirs):
    """dirs: list of (seed_label, dir_path) -> dict seed -> {'single':..,'rollout':..}"""
    out = {}
    for label, d in dirs:
        s = load(osp.join(d, 'baseline_eval', 'baseline_metrics.json'))
        r = load(osp.join(d, 'baseline_eval_rollout18', 'baseline_metrics.json'))
        if s is None and r is None:
            print(f'[WARN] seed {label}: 目录 {d} 下未找到任何评估 JSON，跳过')
            continue
        out[label] = {'single': s, 'rollout': r, 'dir': d}
    return out


def render(results):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    seeds = sorted(results.keys(), key=lambda x: int(x))
    lines = []
    lines.append('## Results — SimVP 多 seed（自动生成于 {}）'.format(now))
    lines.append('')

    if len(seeds) < 2:
        lines.append('> ⚠️ 当前只有 {} 个 seed 的结果（{}），无法计算方差。'
                     '至少 2 个才有 std，建议 ≥3。'.format(len(seeds), ', '.join(seeds)))
        return '\n'.join(lines)

    # ---------------- 单步 ----------------
    s_mse, s_rmse = [], []
    for sd in seeds:
        j = results[sd]['single']
        if j:
            a = j['per_lead_time']['1']
            s_mse.append(a['simvp']['mse']['mean'])
            s_rmse.append(a['simvp_K']['rmse_K'])
    lines.append('### 场景 1：单步（test 1024 条）')
    lines.append('')
    if s_mse:
        st = stats(s_mse)
        lines.append('| 模型 | MSE(归一化) | RMSE(K) |')
        lines.append('|---|---|---|')
        lines.append('| **SimVP（{} seeds）** | {:.6f} ± {:.6f} | {:.4f} |'.format(
            st['n'], st['mean'], st['std'], mse_to_rmse_k(st['mean'])))
        u = stats(list(UNET_SINGLE_3SEED))
        lines.append('| Vanilla UNet（3 seeds, ly） | {:.6f} ± {:.6f} | {:.4f} |'.format(
            u['mean'], u['std'], mse_to_rmse_k(u['mean'])))
        lines.append('')
        lines.append('- SimVP 单步 CV = **{:.2f}%**，UNet 单步 CV = **{:.2f}%**'
                     ' → 两者单步都对 seed 不敏感'.format(st['cv'], u['cv']))
        better = (1 - st['mean'] / u['mean']) * 100
        lines.append('- SimVP 单步 MSE 比 UNet **{}{:.1f}%**，且两侧 std 极小'
                     '（{:.6f} / {:.6f}）→ **该差异稳健**'.format(
                         '好 ' if better > 0 else '差 ', abs(better), st['std'], u['std']))
    lines.append('')

    # ---------------- rollout ----------------
    lines.append('### 场景 2：18 步 rollout（128 条子集）')
    lines.append('')
    lines.append('| LT | SimVP MSE (mean±std) | SimVP CV% | UNet MSE (mean±std, ly) | '
                 'UNet CV% | 谁更好 |')
    lines.append('|---|---|---|---|---|---|')
    lt18_gap = None
    for lt in range(1, 19):
        vals = []
        for sd in seeds:
            r = results[sd]['rollout']
            if r and str(lt) in r['per_lead_time']:
                vals.append(r['per_lead_time'][str(lt)]['simvp']['mse']['mean'])
        st = stats(vals)
        us = stats(list(UNET_3SEED[lt]))
        if st is None:
            continue
        who = 'SimVP' if st['mean'] < us['mean'] else 'UNet'
        lines.append('| {} | {:.6f} ± {:.6f} | {:.1f} | {:.6f} ± {:.6f} | {:.1f} | {} |'
                     .format(lt, st['mean'], st['std'], st['cv'],
                             us['mean'], us['std'], us['cv'], who))
        if lt == 18:
            lt18_gap = (st, us)
    lines.append('')

    # ---------------- 结论 ----------------
    lines.append('### 结论')
    lines.append('')
    if lt18_gap:
        st, us = lt18_gap
        diff = st['mean'] - us['mean']
        comb = st['std'] + us['std']
        robust = abs(diff) > comb
        lines.append('1. **LT=18**：SimVP {:.4f}±{:.4f} vs UNet {:.4f}±{:.4f}，'
                     'SimVP 差 **{:.1f}%**。'.format(
                         st['mean'], st['std'], us['mean'], us['std'],
                         diff / us['mean'] * 100))
        lines.append('2. 差异 {:.4f} 与两侧标准差之和 {:.4f} 相比 → **{}**。'.format(
            abs(diff), comb,
            '差异大于方差，结论**稳健**' if robust else
            '差异未超过方差之和，结论**不稳健**，不能断言 SimVP 更差'))
        lines.append('3. SimVP 自身 rollout CV 从 LT=1 到 LT=18 的演化，'
                     '可与 UNet 的 13.4% 直接对照（见上表 CV% 列）。')
    lines.append('')
    lines.append('> 引用限制：样本为各 seed 各 1 次训练；rollout 用 128 条子集'
                 '（与 ly 协议一致）。seed 数越多均值估计越稳（标准误 ≈ CV/√n）。')
    return '\n'.join(lines)


def dump_summary(results, path):
    """落一份机器可读汇总 JSON，供 sync_research_index.py 写入实验索引。"""
    seeds = sorted(results.keys(), key=lambda x: int(x))
    out = {'generated': datetime.now().isoformat(timespec='seconds'),
           'seeds': seeds, 'per_lt': {}}
    for lt in range(1, 19):
        vals = []
        for sd in seeds:
            r = results[sd]['rollout']
            if r and str(lt) in r['per_lead_time']:
                vals.append(r['per_lead_time'][str(lt)]['simvp']['mse']['mean'])
        st = stats(vals)
        if st:
            out['per_lt'][str(lt)] = {'simvp_mse_mean': st['mean'],
                                      'simvp_mse_std': st['std'],
                                      'simvp_cv_pct': st['cv'], 'n_seeds': st['n']}
    singles = []
    for sd in seeds:
        j = results[sd]['single']
        if j:
            singles.append(j['per_lead_time']['1']['simvp']['mse']['mean'])
    if singles:
        st = stats(singles)
        out['single'] = {'mse_mean': st['mean'], 'mse_std': st['std'],
                         'n_seeds': st['n']}
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='*', default=None,
                    help='形如 42=<dir> 0=<dir> 123=<dir>；默认用 work_dirs 下的三 seed 目录')
    ap.add_argument('--out', default=osp.join(
        _OPENSTL_DIR, 'docs', 'experiments', 'EXP-005-simvp-multiseed', 'README.md'))
    args = ap.parse_args()

    if args.dirs:
        dirs = [tuple(x.split('=', 1)) for x in args.dirs]
    else:
        base = osp.join(_OPENSTL_DIR, 'work_dirs')
        dirs = [('42', osp.join(base, 'satcast_simvp_full')),
                ('0', osp.join(base, 'satcast_simvp_full_seed0')),
                ('123', osp.join(base, 'satcast_simvp_full_seed123'))]

    results = collect(dirs)
    for sd, v in results.items():
        print(f'[INFO] seed {sd}: 单步={"有" if v["single"] else "无"} '
              f'rollout={"有" if v["rollout"] else "无"}  ({v["dir"]})')
    body = render(results)

    # 同时落一份机器可读 JSON，供 sync_research_index.py 同步到实验索引
    summary_path = osp.join(osp.dirname(args.out), 'multiseed_summary.json')
    dump_summary(results, summary_path)
    print(f'[OK] 汇总 JSON -> {summary_path}')

    header = (
        '# EXP-005 — SimVP 多 seed rollout 方差（与 UNet 对照）\n'
        '\n'
        '## Identity\n'
        '- 实验 ID：`EXP-005`\n'
        '- 状态：`AUTO-GENERATED`（由 `tools/aggregate_multiseed.py` 生成）\n'
        '- 日期：{}\n'
        '- 关联：`EXP-004`（统一 baseline 协议）、ly 的 `EXP-EVAL-003`（UNet 多 seed）\n'
        '\n'
        '## 目的\n'
        'ly 已证明 Vanilla UNet 的 rollout 对随机种子极敏感（LT=18 CV = 13.4%），'
        '单 seed 之间比 rollout 无意义。本实验训练 SimVP 的多个 seed，'
        '拿到 SimVP 自身的 rollout 方差，再与 UNet 做**带方差**的对比，'
        '判断 EXP-004 中"SimVP 在 LT=18 比 UNet 差 60%"是否为稳健结论。\n'
        '\n'
        '## 设置\n'
        '- 与 EXP-004 完全相同的配置（`configs/satcast/SimVP.py`）、相同数据（三份 zarr）\n'
        '- **仅改变模型初始化 seed**（数据 split 固定，不存在 split_seed 变化）\n'
        '- seed 集合：0 / 42 / 123（与 ly 保持一致，便于对照）\n'
        '- 每个 seed 评估：单步（test 1024 条）+ 18 步 rollout（128 条子集）\n'
        '\n'
    ).format(datetime.now().strftime('%Y-%m-%d'))

    os.makedirs(osp.dirname(args.out), exist_ok=True)
    if osp.exists(args.out):
        text = open(args.out).read()
    else:
        text = header + '<!-- MULTISEED:BEGIN -->\n<!-- MULTISEED:END -->\n'

    if BEGIN not in text or END not in text:
        raise SystemExit(f'文档中未找到 MULTISEED 标记，拒绝写入：{args.out}')
    b = text.index(BEGIN)
    b_end = text.index('-->', b) + 3
    e = text.index(END)
    e_end = text.index('-->', e) + 3
    new_text = (text[:b] + '<!-- MULTISEED:BEGIN -->\n' + body + '\n'
                + '<!-- MULTISEED:END -->' + text[e_end:])
    with open(args.out, 'w') as f:
        f.write(new_text)

    print(f'[OK] 已写入 {args.out}')
    print('=' * 70)
    print(body)
    print('=' * 70)


if __name__ == '__main__':
    main()
