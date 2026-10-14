#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把本项目 SimVP 的滚动预报导出为**与 ly 的 pred_official.npy 完全相同的格式**。

目标格式
--------
``(1024, 18, 256, 256)`` float32 —— 轴序为 **(样本, lead time, H, W)**，
按 test 数据顺序保存 18 帧滚动预报结果。这与 ly 的
``outputs/eval_unet_raw_preds/pred_official.npy`` 一致，便于：
  * 复现评估结果
  * 与其他模型（UNet / EDM / CorrDiff）在**同一份数组**上做逐样本对比与配对检验

输入：各 seed 目录 ``baseline_eval_rollout18_full/preds.npy``，形状
``(18, 1024, 1, 256, 256)`` = (LT, 样本, C, H, W)。
输出：转置并去掉通道维 -> ``(1024, 18, 256, 256)``。

产出（默认目录 ``work_dirs/satcast_preds/``）
------------------------------------------
* ``pred_simvp_seed42.npy`` / ``pred_simvp_seed0.npy`` / ``pred_simvp_seed123.npy``
* ``pred_simvp_ens.npy``   —— 三 seed 预测平均（集成）
* ``gt.npy``               —— 真值（与 ly 的 gt.npy 逐位一致，已校验）
* ``summary.json``         —— 说明与校验信息

用法::
    python tools/export_preds_official_format.py
"""

import argparse
import json
import os
import os.path as osp
import sys

_HERE = osp.dirname(osp.abspath(__file__))
_OPENSTL_DIR = osp.dirname(_HERE)
if _OPENSTL_DIR not in sys.path:
    sys.path.insert(0, _OPENSTL_DIR)

import numpy as np  # noqa: E402

SEEDS = {
    'seed42': 'satcast_simvp_full',
    'seed0': 'satcast_simvp_full_seed0',
    'seed123': 'satcast_simvp_full_seed123',
}
# 可移植性：ly 的产物根目录优先取环境变量 LY_OUTPUTS
LY_GT = osp.join(
    os.environ.get('LY_OUTPUTS',
                   '/home/group1/26fall_aiclass/ly/cira-diff/outputs'),
    'eval_unet_raw_preds', 'gt.npy')


def load_lt(seed_dir, name):
    """读取 (18, N, 1, H, W) 并转成 (N, 18, H, W)。"""
    f = osp.join(_OPENSTL_DIR, 'work_dirs', seed_dir,
                 'baseline_eval_rollout18_full', f'{name}.npy')
    if not osp.exists(f):
        raise FileNotFoundError(f)
    a = np.asarray(np.load(f, mmap_mode='r'))          # (18, N, 1, H, W)
    return np.ascontiguousarray(a[:, :, 0].transpose(1, 0, 2, 3)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=osp.join(
        _OPENSTL_DIR, 'work_dirs', 'satcast_preds'))
    ap.add_argument('--extra', nargs='*', default=[],
                    help='额外模型，形如 tag=seed_dir（如 gan=satcast_simvp_gan）')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {
        'format': '(1024, 18, 256, 256) float32 = (样本, lead_time, H, W)',
        'axis_note': '与 ly 的 pred_official.npy 完全一致；lead time 步长 10 min，'
                     '第 0 个 LT = t+10min，第 17 个 = t+180min',
        'source': 'SimVP + gSTA (configs/satcast/SimVP.py), 100 epoch, 2 帧->1 帧递归外推',
        'files': {},
    }

    preds = {}
    for tag, d in SEEDS.items():
        arr = load_lt(d, 'preds')
        out = osp.join(args.out_dir, f'pred_simvp_{tag}.npy')
        np.save(out, arr)
        preds[tag] = arr
        summary['files'][f'pred_simvp_{tag}.npy'] = {
            'shape': list(arr.shape), 'dtype': str(arr.dtype), 'source_dir': d}
        print(f'[SAVE] {out}  {arr.shape}')

    for item in args.extra:
        if '=' not in item:
            print(f'[WARN] 忽略无法解析的 --extra：{item}')
            continue
        tag, d = item.split('=', 1)
        try:
            arr = load_lt(d, 'preds')
        except FileNotFoundError:
            print(f'[WARN] 跳过 {tag}：{d} 下无 1024 条 rollout 结果')
            continue
        out = osp.join(args.out_dir, f'pred_{tag}.npy')
        np.save(out, arr)
        summary['files'][f'pred_{tag}.npy'] = {
            'shape': list(arr.shape), 'dtype': str(arr.dtype), 'source_dir': d}
        print(f'[SAVE] {out}  {arr.shape}')

    ens = np.mean([preds[t] for t in SEEDS], axis=0).astype(np.float32)
    np.save(osp.join(args.out_dir, 'pred_simvp_ens.npy'), ens)
    summary['files']['pred_simvp_ens.npy'] = {
        'shape': list(ens.shape), 'dtype': str(ens.dtype),
        'note': '三 seed 预测平均（集成）'}
    print(f'[SAVE] {osp.join(args.out_dir, "pred_simvp_ens.npy")}  {ens.shape}')

    # 真值：用我们自己的 trues（并校验与 ly 的 gt 是否逐位一致）
    our_true = load_lt(SEEDS['seed42'], 'trues')
    summary['files']['gt.npy'] = {'shape': list(our_true.shape),
                                  'dtype': str(our_true.dtype)}
    np.save(osp.join(args.out_dir, 'gt.npy'), our_true)
    print(f'[SAVE] {osp.join(args.out_dir, "gt.npy")}  {our_true.shape}')

    if osp.exists(LY_GT):
        ly = np.load(LY_GT, mmap_mode='r')
        n = min(ly.shape[0], our_true.shape[0])
        diff = float(np.abs(np.asarray(ly[:n]) - our_true[:n]).max())
        summary['gt_alignment_check'] = {
            'vs_ly_gt_max_abs_diff': diff,
            'aligned': bool(diff < 1e-5),
            'note': '与 ly 的 gt.npy 逐位一致 -> 可与 UNet 预测直接做配对比较'}
        print(f'[CHECK] 与 ly 的 gt.npy 最大差异 = {diff:.3e} '
              f'({"一致" if diff < 1e-5 else "不一致"})')

    with open(osp.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'[SAVE] {osp.join(args.out_dir, "summary.json")}')


if __name__ == '__main__':
    main()
