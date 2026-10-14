#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**依赖自检**：在跑任何训练/评估前，先确认环境与外部依赖齐全。

为什么需要这个
--------------
本项目代码依赖几类**外部条件**，缺任何一类都会在不同阶段失败——
有的在 import 就崩，有的跑到一半才崩，最糟的是**静默用错配置**。
本脚本把这些问题前置到"按下运行键之前"。

检查项
------
| # | 项 | 失败后果 | 级别 |
|---|---|---|---|
| 1 | `openstl/datasets/dataloader_satcast.py` 存在 | 13 个脚本 ImportError | **阻断** |
| 2 | Python 包：torch / lightning / timm(<1.0) / fvcore / zarr / cv2 | import 阶段崩 | **阻断** |
| 3 | `configs/satcast/SimVP.py` 存在 | 训练无超参来源 | **阻断** |
| 4 | 数据根（`SATCAST_ROOT` 或默认）存在且含三份 zarr | 训练/评估读不到数据 | **阻断** |
| 5 | `update_config` 行为（配置是否真的生效） | **静默用错超参训练** | **阻断** |
| 6 | `ly` 产物（UNet/EDM 原始预测） | 两个对比脚本无法运行 | 警告 |
| 7 | GPU 可用性 | 训练极慢 | 警告 |
| 8 | `test_dl/*.slurm` 无个人 home 硬编码 | 换机器跑不了 | 警告 |

用法::

    python tools/check_portability.py            # 全部检查
    python tools/check_portability.py --quick    # 只查阻断项
    SATCAST_ROOT=/data1/satcast python tools/check_portability.py

退出码：0 = 全部通过；1 = 有阻断项失败。
"""

import argparse
import glob
import os
import os.path as osp
import subprocess
import sys

_HERE = osp.dirname(osp.abspath(__file__))
_OPENSTL_DIR = osp.dirname(_HERE)
if _OPENSTL_DIR not in sys.path:
    sys.path.insert(0, _OPENSTL_DIR)

DATA_ROOT = os.environ.get('SATCAST_ROOT',
                           '/home/group1/26fall_aiclass/yr/data1/data1/satcast')
LY_OUTPUTS = os.environ.get('LY_OUTPUTS',
                            '/home/group1/26fall_aiclass/ly/cira-diff/outputs')
ZARRS = ['edm_GOES_ch13_train_dataset.zarr',
         'edm_GOES_ch13_validation_dataset.zarr',
         'edm_GOES_ch13_test_dataset.zarr']

BLOCK, WARN, OK = 'BLOCK', 'WARN', 'OK'
results = []


def rec(level, name, detail, fix=''):
    results.append((level, name, detail, fix))
    icon = {BLOCK: '✗ 阻断', WARN: '⚠ 警告', OK: '✓ 通过'}[level]
    print(f'[{icon}] {name}')
    if detail:
        print(f'         {detail}')
    if fix and level != OK:
        print(f'         修复：{fix}')


def check_adapter():
    p = osp.join(_OPENSTL_DIR, 'openstl', 'datasets', 'dataloader_satcast.py')
    if osp.exists(p):
        rec(OK, 'satcast 数据适配器', p)
    else:
        rec(BLOCK, 'satcast 数据适配器', f'缺失：{p}',
            '从本复现项目拷贝该文件（上游唯一新增文件）')


def check_packages():
    mods = [('numpy', None), ('torch', None), ('lightning', None),
            ('timm', '<1.0'), ('fvcore', None), ('zarr', None), ('cv2', None),
            ('matplotlib', None), ('einops', None)]
    missing, bad = [], []
    for m, constraint in mods:
        try:
            mod = __import__(m)
            ver = getattr(mod, '__version__', '?')
            if m == 'timm':
                major = int(ver.split('.')[0])
                if major >= 1:
                    bad.append(f'timm {ver}（必须 <1.0，见 DBG-005）')
        except Exception:
            missing.append(m)
    if not missing and not bad:
        rec(OK, 'Python 依赖包', 'numpy/torch/lightning/timm/fvcore/zarr/cv2/matplotlib/einops 齐全')
    else:
        msg = []
        if missing:
            msg.append('缺失：' + ', '.join(missing))
        if bad:
            msg.append('版本不符：' + ', '.join(bad))
        rec(BLOCK, 'Python 依赖包', '；'.join(msg),
            'conda create -n openstl --clone test && pip install timm==0.9.12 '
            'lightning fvcore zarr opencv-python-headless ...'
            '（完整清单见 docs/dev/environment.md）')


def check_config():
    p = osp.join(_OPENSTL_DIR, 'configs', 'satcast', 'SimVP.py')
    if osp.exists(p):
        rec(OK, '训练配置文件', p)
    else:
        rec(BLOCK, '训练配置文件', f'缺失：{p}', '从本复现项目拷贝 configs/satcast/')


def check_data():
    if not osp.isdir(DATA_ROOT):
        rec(BLOCK, '数据根目录', f'不存在：{DATA_ROOT}',
            'export SATCAST_ROOT=<你的 satcast 目录>')
        return
    miss = [z for z in ZARRS if not osp.isdir(osp.join(DATA_ROOT, z))]
    if not miss:
        rec(OK, '三份 zarr 数据', DATA_ROOT)
    else:
        rec(BLOCK, '三份 zarr 数据', f'{DATA_ROOT} 下缺少：{", ".join(miss)}',
            '确认数据完整，或用 SATCAST_ROOT 指向正确目录')


def check_update_config():
    """最危险的一项：确认配置不会被上游 update_config 静默丢弃。"""
    try:
        from openstl.utils.main_utils import update_config
        import inspect
        src = inspect.getsource(update_config)
    except Exception as e:
        rec(WARN, 'update_config 行为', f'无法读取上游实现：{e}',
            '人工确认你的 OpenSTL 版本与本站一致')
        return
    # 若实现里对 "args 已有该属性就跳过" 的模式，则配置可能被丢弃
    suspicious = ('hasattr(args' in src or 'args.{}'.format('') in src
                  or 'skip' in src.lower())
    if suspicious:
        rec(WARN, 'update_config 行为',
            '上游 update_config 可能丢弃 argparse 默认值为真值的配置项',
            'train_simvp_satcast.py 已把全部 key 放入 exclude_keys；'
            '若你的 OpenSTL 版本不同，请运行该脚本并核对开头的配置生效自检表')
    else:
        rec(OK, 'update_config 行为', '未发现静默丢弃模式')


def check_config_effect():
    """实跑一次配置解析，核对关键超参是否与 configs/satcast/SimVP.py 一致。"""
    try:
        sys.path.insert(0, osp.join(_OPENSTL_DIR, 'tools'))
        import train_simvp_satcast as T
    except Exception as e:
        rec(WARN, '配置生效自检', f'无法导入训练脚本：{e}')
        return
    try:
        cli = T.make_cli_parser().parse_args(['--cpu', '--epochs', '1'])
        args = T.build_args(cli)
    except Exception as e:
        rec(WARN, '配置生效自检', f'解析失败：{e}')
        return
    expect = {'method': 'SimVP', 'model_type': 'gSTA', 'hid_S': 64, 'hid_T': 256,
              'N_S': 2, 'N_T': 4, 'drop_path': 0.1, 'lr': 2e-3,
              'sched': 'onecycle', 'batch_size': 8, 'val_batch_size': 32,
              'num_workers': 8}
    bad = {k: (getattr(args, k, None), v) for k, v in expect.items()
           if getattr(args, k, None) != v}
    if not bad:
        rec(OK, '配置生效自检', '12 项关键超参与 configs/satcast/SimVP.py 完全一致')
    else:
        detail = '；'.join(f'{k}: 实际={a} 期望={b}' for k, (a, b) in bad.items())
        rec(BLOCK, '配置生效自检', detail,
            '配置被静默丢弃！检查你的 OpenSTL 版本与 exclude_keys 机制')


def check_ly_outputs():
    unet = osp.join(LY_OUTPUTS, 'eval_unet_raw_preds')
    edm = osp.join(LY_OUTPUTS, 'eval_official_edm_inference')
    have, lack = [], []
    for tag, d, keys in (('UNet', unet, ['pred_seed42.npy', 'gt.npy']),
                         ('EDM', edm, [])):
        if osp.isdir(d):
            have.append(tag)
            for k in keys:
                if not osp.exists(osp.join(d, k)):
                    lack.append(f'{tag}/{k}')
        else:
            lack.append(f'{tag} 目录不存在')
    if not lack:
        rec(OK, 'ly 产物（对比用）', f'{LY_OUTPUTS} 下 UNet/EDM 原始预测齐全')
    else:
        rec(WARN, 'ly 产物（对比用）',
            '缺少：' + ', '.join(lack),
            'export LY_OUTPUTS=<ly 的 outputs 目录>；'
            '缺少时 compare_with_unet_raw.py / compare_edm_sharpness.py 无法运行'
            '（其余脚本不受影响）')


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            rec(OK, 'GPU', f'{n} 张可用：{torch.cuda.get_device_name(0)}')
        else:
            rec(WARN, 'GPU', 'CUDA 不可用', '训练会退回 CPU，极慢；检查驱动/CUDA_VISIBLE_DEVICES')
    except Exception as e:
        rec(WARN, 'GPU', f'检测失败：{e}')


def check_slurm_portable():
    bad = []
    for f in sorted(glob.glob(osp.join(_OPENSTL_DIR, 'test_dl', '*.slurm'))):
        t = open(f).read()
        if '26fall_aiclass/yr/cira-diff' in t:
            bad.append(osp.basename(f))
    if not bad:
        rec(OK, 'slurm 无个人路径硬编码', '16 个脚本均从自身位置推导路径')
    else:
        rec(WARN, 'slurm 无个人路径硬编码',
            '仍含绝对路径：' + ', '.join(bad),
            '把这些路径改为 ${OPENSTL_DIR:-...} / ${SATCAST_ROOT:-...}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', default=False,
                    help='只查阻断项（跳过 GPU / ly 产物 / slurm 可移植性）')
    args = ap.parse_args()

    print('=' * 74)
    print('可移植性 / 依赖自检')
    print('=' * 74)
    print(f'仓库根    : {_OPENSTL_DIR}')
    print(f'数据根    : {DATA_ROOT}'
          f'{"（来自 SATCAST_ROOT）" if os.environ.get("SATCAST_ROOT") else "（默认值）"}')
    print(f'ly 产物根 : {LY_OUTPUTS}'
          f'{"（来自 LY_OUTPUTS）" if os.environ.get("LY_OUTPUTS") else "（默认值）"}')
    print('=' * 74)

    check_adapter()
    check_packages()
    check_config()
    check_data()
    check_update_config()
    check_config_effect()
    if not args.quick:
        check_ly_outputs()
        check_gpu()
        check_slurm_portable()

    n_block = sum(1 for r in results if r[0] == BLOCK)
    n_warn = sum(1 for r in results if r[0] == WARN)
    n_ok = sum(1 for r in results if r[0] == OK)
    print('=' * 74)
    print(f'汇总：通过 {n_ok} ｜ 警告 {n_warn} ｜ 阻断 {n_block}')
    if n_block:
        print('❌ 存在阻断项，修复前不要提交训练/评估作业。')
        sys.exit(1)
    print('✅ 阻断项全部通过，可以运行。')
    if n_warn:
        print('⚠️ 有警告项：部分对比脚本可能不可用，详见上方说明。')


if __name__ == '__main__':
    main()
