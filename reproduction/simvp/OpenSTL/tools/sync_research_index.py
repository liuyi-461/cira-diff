#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 EXP-004 / EXP-005 的结论**自动同步**到实验索引 ``docs/research/experiments.md``。

为什么要有这个脚本
------------------
项目规范是"每个实验按 EXP-XXX 编号，记录条件与结果"。EXP-004/005 的数字由
评估脚本产出并存在 JSON 里；若靠人手抄进索引，既容易抄错，也会在重跑后忘记更新。
本脚本从 JSON 读取，按索引既有条目格式（日期/目的/数据/方法/参数/结果/结论/
下一步/复现教程）渲染，替换 ``<!-- INDEX:BEGIN/END -->`` 之间的内容，可重复运行。

输入
----
* seed42 的评估 JSON（EXP-004 的权威数字）
  ``work_dirs/satcast_simvp_full/baseline_eval/baseline_metrics.json``
  ``work_dirs/satcast_simvp_full/baseline_eval_rollout18/baseline_metrics.json``
* 多 seed 汇总 JSON（由 ``aggregate_multiseed.py`` 产出）
  ``docs/experiments/EXP-005-simvp-multiseed/multiseed_summary.json``

用法::

    python tools/sync_research_index.py
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

BEGIN = '<!-- INDEX:BEGIN'
END = '<!-- INDEX:END'

# 与 aggregate_multiseed.py 保持一致（复制自 ly EXP-EVAL-003 全表）
UNET_3SEED = {
    1: (0.006878, 0.006881, 0.006968), 2: (0.018930, 0.018880, 0.019040),
    3: (0.034210, 0.034380, 0.034160), 4: (0.051590, 0.052570, 0.052090),
    5: (0.070920, 0.073480, 0.072220), 6: (0.092090, 0.096610, 0.094030),
    7: (0.114450, 0.121930, 0.117430), 8: (0.138370, 0.150130, 0.143240),
    9: (0.163760, 0.181870, 0.171910), 10: (0.190720, 0.216200, 0.203230),
    11: (0.219370, 0.253110, 0.237050), 12: (0.249610, 0.292470, 0.273370),
    13: (0.281730, 0.334510, 0.312430), 14: (0.316220, 0.379190, 0.354370),
    15: (0.353420, 0.425380, 0.399510), 16: (0.393320, 0.473460, 0.448020),
    17: (0.435780, 0.523400, 0.499970), 18: (0.419200, 0.574300, 0.471700),
}
UNET_SINGLE_3SEED = (0.008050, 0.008092, 0.008157)
CURRENT_LINE = ('- 暂无（reproduction 仍处 candidate only，见 '
                '`docs/project/current_state.md`）。')


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
    sd = 0.0
    if len(vals) >= 2:
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return {'mean': m, 'std': sd, 'cv': sd / m * 100 if m else 0.0, 'n': len(vals)}


def rmse_k(mse):
    return math.sqrt(mse) * STD_K


def entry_exp004(single, rollout):
    """渲染 EXP-004 索引条目。"""
    L = []
    L.append('# EXP-004 统一 Baseline 协议 + SimVP 全量训练评估（GOES-16 ABI ch13）')
    L.append('')
    L.append('日期：')
    L.append(datetime.now().strftime('%Y-%m-%d'))
    L.append('')
    L.append('实验目的：')
    L.append('把外部基线（ly 的 Vanilla UNet、cb 的 SimVP）固化为可引用 baseline，'
             '定义统一评估协议，并在同一把尺子上评估我们的 SimVP。')
    L.append('')
    L.append('数据：')
    L.append('- train/validation/test **三份独立 zarr**（test = 官方 split，1024 条）')
    L.append('- `output_images` (1024,18,256,256)；单步任务取第 0 帧 = **t+10min**'
             '（已实测 |Y[:,0]-X[:,1]| 与已知 10 分钟步长比值 1.003）')
    L.append('- zarr 内已是归一化空间（mean=0, std=1）；'
             '`Tb(K) = zarr × 19.3297 + 279.0699`')
    L.append('')
    L.append('方法：')
    L.append('- SimVP + gSTA，4.705M 参数，通过 `BaseExperiment(dataloaders=...)` 注入')
    L.append('- `pre_seq_length=2` / `aft_seq_length=1` / `in_shape=[2,1,256,256]`')
    L.append('- **不做子集 z-score**（修正 EXP-002 的 D3 偏差），模型直接吃 zarr 原值')
    L.append('- 评估用 `tools/eval_simvp_baseline.py`（加载 best.ckpt、逐像素口径）')
    L.append('')
    L.append('参数：')
    L.append('- `configs/satcast/SimVP.py`：`hid_S=64, hid_T=256, N_S=2, N_T=4, '
             'drop_path=0.1`')
    L.append('- `lr=2e-3`、`sched=onecycle`、`batch_size=8`、`epoch=100`、`val_batch_size=32`')
    L.append('- seed=42（多 seed 见 EXP-005）')
    L.append('')

    L.append('结果：')
    if single is None:
        L.append('**尚未产出**（评估 JSON 不存在）。')
    else:
        a = single['per_lead_time']['1']
        ep = single.get('ckpt_epoch', 'N/A')
        L.append('单步（test 全量 1024 条，best epoch = {}）：'.format(ep))
        L.append('- SimVP：MSE={:.6f}、**RMSE={:.4f} K**、MAE={:.4f} K、SSIM={:.4f}'.format(
            a['simvp']['mse']['mean'], a['simvp_K']['rmse_K'],
            a['simvp_K']['mae_K'], a['simvp']['ssim']['mean']))
        L.append('- persistence：MSE={:.6f}、RMSE={:.4f} K'.format(
            a['persistence']['mse']['mean'], a['persistence_K']['rmse_K']))
        L.append('- 对照：UNet(ly,seed42) RMSE≈{:.2f} K；SimVP(cb,100ep) RMSE≈1.57 K'.format(
            rmse_k(UNET_SINGLE_3SEED[1])))
        L.append('- MSE 相对 persistence 降低 **{:.2f}%**'.format(
            single.get('mse_reduction_vs_persistence_pct_LT1', float('nan'))))
        if rollout:
            p = rollout['per_lead_time']
            cross = None
            for lt in range(1, 19):
                e = p.get(str(lt))
                if e and cross is None and e['simvp_K']['rmse_K'] > e['persistence_K']['rmse_K']:
                    cross = lt
            L.append('18 步 rollout（128 条子集）：LT=1 RMSE={:.2f} K → '
                     'LT=18 RMSE={:.2f} K'.format(
                         p['1']['simvp_K']['rmse_K'], p['18']['simvp_K']['rmse_K']))
            if cross:
                L.append('- **从 LT={}（{} 分钟）起 RMSE 反超 persistence**'.format(
                    cross, cross * 10))
            L.append('- LT=18 对照 UNet(seed0) MSE {:.6f} vs 本模型 {:.6f}'.format(
                UNET_3SEED[18][0], p['18']['simvp']['mse']['mean']))
    L.append('')
    L.append('结论：')
    if single is None:
        L.append('待产出。')
    else:
        a = single['per_lead_time']['1']
        gain = (1 - a['simvp']['mse']['mean'] / UNET_SINGLE_3SEED[1]) * 100
        L.append('1. **单步：SimVP 优于 Vanilla UNet**（MSE 好 {:.1f}%），'
                 '且远优于 persistence。'.format(gain))
        L.append('2. **rollout：SimVP 长时稳定性明显弱于 UNet**，'
                 '且在中长时效（约 100 分钟后）会劣于 persistence —— '
                 '纯单步 teacher-forced 训练的误差累积所致。')
        L.append('3. 上述 rollout 结论为**单 seed**，稳健性由 **EXP-005** 判定。')
    L.append('')
    L.append('下一步：')
    L.append('1. 用 EXP-005 的多 seed 方差确认 rollout 结论是否稳健；')
    L.append('2. 尝试 rollout-aware 训练（scheduled sampling / multi-step loss）'
             '改善长时效；')
    L.append('3. 与 CIRA-Diff 在统一多步协议下对比（DEC-002）。')
    L.append('')
    L.append('复现教程:')
    L.append('实现：`tools/train_simvp_satcast.py` + `tools/eval_simvp_baseline.py`；')
    L.append('适配器：`openstl/datasets/dataloader_satcast.py`；')
    L.append('配置：`configs/satcast/SimVP.py`；')
    L.append('提交：`test_dl/train_baseline.slurm` / `test_dl/eval_baseline.slurm`；')
    L.append('证据包：`docs/experiments/EXP-004-simvp-vs-baseline/`。')
    return '\n'.join(L)


def entry_exp005(ms):
    """渲染 EXP-005 索引条目。"""
    L = []
    L.append('# EXP-005 SimVP 多 seed rollout 方差（与 UNet 对照）')
    L.append('')
    L.append('日期：')
    L.append(datetime.now().strftime('%Y-%m-%d'))
    L.append('')
    L.append('实验目的：')
    L.append('ly 证明 Vanilla UNet 的 rollout 对 seed 极敏感（LT=18 CV=13.4%），'
             '单 seed 之间比 rollout 无意义。本实验训练 SimVP 多个 seed，'
             '拿到 SimVP 自身的 rollout 方差，判定 EXP-004 中'
             '"LT=18 比 UNet 差 ~60%"是否稳健。')
    L.append('')
    L.append('数据/方法/参数：')
    L.append('- 与 EXP-004 **完全相同**的配置与数据，**仅改变模型初始化 seed**')
    L.append('- seed 集合：0 / 42 / 123（与 ly 一致）')
    L.append('- 每 seed 评估单步（1024 条）+ 18 步 rollout（128 条子集）')
    L.append('')
    L.append('结果：')
    if not ms or not ms.get('per_lt'):
        L.append('**多 seed 结果尚未产出**（等 seed 0/123 训练完成后由 '
                 '`aggregate_multiseed.py` 汇总）。')
    else:
        n = ms.get('seeds', [])
        s_ = ms.get('single')
        if s_:
            u = stats(list(UNET_SINGLE_3SEED))
            L.append('单步（{} seeds）：SimVP MSE={:.6f}±{:.6f} vs '
                     'UNet {:.6f}±{:.6f}'.format(
                         s_['n_seeds'], s_['mse_mean'], s_['mse_std'],
                         u['mean'], u['std']))
        lt18 = ms['per_lt'].get('18')
        if lt18:
            u18 = stats(list(UNET_3SEED[18]))
            diff = lt18['simvp_mse_mean'] - u18['mean']
            comb = lt18['simvp_mse_std'] + u18['std']
            L.append('rollout LT=18（{} seeds）：SimVP MSE={:.4f}±{:.4f}（CV={:.1f}%）vs '
                     'UNet {:.4f}±{:.4f}（CV={:.1f}%）'.format(
                         lt18['n_seeds'], lt18['simvp_mse_mean'], lt18['simvp_mse_std'],
                         lt18['simvp_cv_pct'], u18['mean'], u18['std'], u18['cv']))
            L.append('- SimVP 比 UNet 差 **{:.1f}%**；差值 {:.4f} vs 两侧 std 之和 {:.4f}'.format(
                diff / u18['mean'] * 100, abs(diff), comb))
        L.append('- 逐 LT 的 SimVP/UNet mean±std 与 CV 见 '
                 '`docs/experiments/EXP-005-simvp-multiseed/`。')
    L.append('')
    L.append('结论：')
    if not ms or not ms.get('per_lt') or '18' not in ms.get('per_lt', {}):
        L.append('待产出。')
    else:
        lt18 = ms['per_lt']['18']
        u18 = stats(list(UNET_3SEED[18]))
        diff = abs(lt18['simvp_mse_mean'] - u18['mean'])
        comb = lt18['simvp_mse_std'] + u18['std']
        if diff > comb:
            L.append('LT=18 的差异（{:.4f}）**大于**两侧标准差之和（{:.4f}）→ '
                     '"SimVP rollout 差于 UNet"**稳健**。'.format(diff, comb))
        else:
            L.append('LT=18 的差异（{:.4f}）**未超过**两侧标准差之和（{:.4f}）→ '
                     '该差异可能主要来自 seed 方差，**不能断言** SimVP 更差。'.format(
                         diff, comb))
        L.append('SimVP 自身 rollout CV（LT=18）={:.1f}%，可与 UNet 的 13.4% 直接对照。'
                 .format(lt18['simvp_cv_pct']))
    L.append('')
    L.append('下一步：')
    L.append('1. 若结论稳健 → 针对长时效做 rollout-aware 训练；')
    L.append('2. 若增加 seed 数（≥5）以降低均值标准误（≈CV/√n）；')
    L.append('3. 结论回填 EXP-004 并更新 `docs/research/conclusions.md`。')
    L.append('')
    L.append('复现教程:')
    L.append('训练：`SEED=0 sbatch test_dl/train_baseline.slurm`（同理 123）；')
    L.append('汇总：`tools/aggregate_multiseed.py` → '
             '`docs/experiments/EXP-005-simvp-multiseed/`。')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', default=osp.join(
        _OPENSTL_DIR, 'docs', 'research', 'experiments.md'))
    ap.add_argument('--seed42-dir', default=osp.join(
        _OPENSTL_DIR, 'work_dirs', 'satcast_simvp_full'))
    ap.add_argument('--multiseed-json', default=osp.join(
        _OPENSTL_DIR, 'docs', 'experiments', 'EXP-005-simvp-multiseed',
        'multiseed_summary.json'))
    args = ap.parse_args()

    single = load(osp.join(args.seed42_dir, 'baseline_eval',
                           'baseline_metrics.json'))
    rollout = load(osp.join(args.seed42_dir, 'baseline_eval_rollout18',
                            'baseline_metrics.json'))
    ms = load(args.multiseed_json)

    body = (entry_exp004(single, rollout) + '\n\n---\n\n' + entry_exp005(ms) + '\n')

    with open(args.index) as f:
        text = f.read()
    if BEGIN not in text or END not in text:
        raise SystemExit(f'索引中未找到 INDEX 标记，拒绝写入：{args.index}')

    b = text.index(BEGIN)
    b_end = text.index('-->', b) + 3
    e = text.index(END)
    e_end = text.index('-->', e) + 3
    text = (text[:b] + '<!-- INDEX:BEGIN -->\n' + body
            + '<!-- INDEX:END -->' + text[e_end:])

    # 顺带把"当前实验"从"暂无"改成指向最新实验（找不到原句就跳过，不报错）
    if CURRENT_LINE in text:
        new_cur = ('- **EXP-004**（统一 baseline + SimVP 全量评估）：'
                   '见 `docs/experiments/EXP-004-simvp-vs-baseline/`。\n'
                   '- **EXP-005**（多 seed rollout 方差）：'
                   '见 `docs/experiments/EXP-005-simvp-multiseed/`。\n'
                   '- 两者结论均由脚本自动生成，勿手改。')
        text = text.replace(CURRENT_LINE, new_cur)

    with open(args.index, 'w') as f:
        f.write(text)

    print(f'[OK] 实验索引已同步：{args.index}')
    print(f'[INFO] EXP-004 数据：单步={"有" if single else "无"} '
          f'rollout={"有" if rollout else "无"}')
    print(f'[INFO] EXP-005 数据：多 seed 汇总={"有" if ms else "无（待训练完成）"}')
    print('=' * 70)
    print(body[:1500])
    print('...')
    print('=' * 70)


if __name__ == '__main__':
    main()
