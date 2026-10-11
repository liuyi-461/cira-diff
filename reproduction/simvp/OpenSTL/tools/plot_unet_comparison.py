#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""绘制 **SimVP vs Vanilla UNet** 的逐时效（lead time）预报性能对比图。

为什么这张图重要
----------------
EXP-004 发现"SimVP 单步优于 UNet，但 rollout 长时明显更弱"，该结论是**单 seed** 的。
ly 已证明 UNet 自身在 LT=18 的 seed 方差 CV=13.4%，因此必须把两边的**多 seed 分布**
画在同一张图上，才能判断差距是真实的还是被 seed 方差淹没。

图的内容（3 个子图，共享横轴 lead time）
--------------------------------------
(a) MSE（归一化空间）vs LT
(b) RMSE（亮温 K）vs LT        —— 主图，含 persistence 基线
(c) SSIM vs LT
每条曲线为「多 seed 均值 ± 1 样本标准差」的带状区域。

数据来源
--------
* SimVP：各 seed 目录下 ``baseline_eval_rollout18/baseline_metrics.json``
* UNet ：ly 的 EXP-EVAL-003 全表（seed0/42/123），内置在下面的常量里；
         其 RMSE(K) 由 MSE 换算：``sqrt(MSE) * STD_K``（与 SimVP 的 global 口径一致）
* persistence：SimVP 评估脚本同时产出（seed 无关，取任一 seed 即可）

用法::

    python tools/plot_unet_comparison.py
"""

import argparse
import csv
import json
import math
import os
import os.path as osp
import sys

_HERE = osp.dirname(osp.abspath(__file__))
_OPENSTL_DIR = osp.dirname(_HERE)
if _OPENSTL_DIR not in sys.path:
    sys.path.insert(0, _OPENSTL_DIR)

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from openstl.datasets.dataloader_satcast import STD_K  # noqa: E402

# ---------------------------------------------------------------------------
# ly 的 Vanilla UNet（128 条子集 rollout），复制自 EXP-EVAL-003 全表
# 顺序：(seed0, seed42, seed123)
# ---------------------------------------------------------------------------
UNET_MSE = {
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
UNET_SSIM = {
    1: (0.9045, 0.9023, 0.9037), 2: (0.7953, 0.7895, 0.7953),
    3: (0.6948, 0.6865, 0.6963), 4: (0.6101, 0.5972, 0.6124),
    5: (0.5409, 0.5221, 0.5437), 6: (0.4841, 0.4587, 0.4867),
    7: (0.4371, 0.4040, 0.4401), 8: (0.3981, 0.3581, 0.4017),
    9: (0.3648, 0.3183, 0.3702), 10: (0.3366, 0.2841, 0.3428),
    11: (0.3120, 0.2546, 0.3187), 12: (0.2907, 0.2281, 0.2976),
    13: (0.2721, 0.2044, 0.2791), 14: (0.2556, 0.1823, 0.2622),
    15: (0.2409, 0.1615, 0.2469), 16: (0.2274, 0.1412, 0.2327),
    17: (0.2147, 0.1219, 0.2192), 18: (0.2411, 0.1034, 0.1942),
}

C_SIMVP = '#1f77b4'
C_UNET = '#d62728'
C_PERSIST = '#7f7f7f'


def load(path):
    if not path or not osp.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    m = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
    return m, sd, len(vals)


def collect_simvp(dirs):
    """返回 {lt: {'mse':[...], 'ssim':[...], 'rmse_k':[...], 'persist_rmse_k':v}}"""
    out = {lt: {'mse': [], 'ssim': [], 'rmse_k': [], 'persist_rmse_k': []}
           for lt in range(1, 19)}
    used = []
    for label, d in dirs:
        j = load(osp.join(d, 'baseline_eval_rollout18', 'baseline_metrics.json'))
        if not j:
            continue
        used.append(label)
        per = j['per_lead_time']
        for lt in range(1, 19):
            e = per.get(str(lt))
            if not e:
                continue
            out[lt]['mse'].append(e['simvp']['mse']['mean'])
            out[lt]['ssim'].append(e['simvp']['ssim']['mean'])
            out[lt]['rmse_k'].append(e['simvp_K']['rmse_K'])
            out[lt]['persist_rmse_k'].append(e['persistence_K']['rmse_K'])
    return out, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='*', default=None,
                    help='形如 42=<dir> 0=<dir> 123=<dir>')
    ap.add_argument('--out', default=osp.join(
        _OPENSTL_DIR, 'docs', 'experiments', 'EXP-005-simvp-multiseed',
        'unet_vs_simvp_rollout.png'))
    args = ap.parse_args()

    if args.dirs:
        dirs = [tuple(x.split('=', 1)) for x in args.dirs]
    else:
        base = osp.join(_OPENSTL_DIR, 'work_dirs')
        dirs = [('42', osp.join(base, 'satcast_simvp_full')),
                ('0', osp.join(base, 'satcast_simvp_full_seed0')),
                ('123', osp.join(base, 'satcast_simvp_full_seed123'))]

    data, used = collect_simvp(dirs)
    n_seeds = len(used)
    print(f'[INFO] 参与统计的 SimVP seeds: {used} (共 {n_seeds} 个)')

    lts = list(range(1, 19))
    s_mse, s_mse_sd = [], []
    s_rmse, s_rmse_sd = [], []
    s_ssim, s_ssim_sd = [], []
    u_mse, u_mse_sd = [], []
    u_rmse, u_rmse_sd = [], []
    u_ssim, u_ssim_sd = [], []
    persist = []

    for lt in lts:
        m, sd, _ = mean_std(data[lt]['mse'])
        s_mse.append(m); s_mse_sd.append(sd)
        m, sd, _ = mean_std(data[lt]['rmse_k'])
        s_rmse.append(m); s_rmse_sd.append(sd)
        m, sd, _ = mean_std(data[lt]['ssim'])
        s_ssim.append(m); s_ssim_sd.append(sd)

        um = list(UNET_MSE[lt])
        m, sd, _ = mean_std(um)
        u_mse.append(m); u_mse_sd.append(sd)
        m, sd, _ = mean_std([math.sqrt(v) * STD_K for v in um])
        u_rmse.append(m); u_rmse_sd.append(sd)
        m, sd, _ = mean_std(list(UNET_SSIM[lt]))
        u_ssim.append(m); u_ssim_sd.append(sd)

        p = data[lt]['persist_rmse_k']
        persist.append(float(np.mean(p)) if p else None)

    # ----------------------------- 画图 -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))

    def panel(ax, sim, sim_sd, unet, unet_sd, extra, ylabel, title, logy=False):
        lt_arr = np.array(lts)
        if all(v is not None for v in sim):
            ax.plot(lt_arr, sim, 'o-', color=C_SIMVP, lw=2,
                    label=f'SimVP ({n_seeds} seed)')
            if n_seeds >= 2:
                ax.fill_between(lt_arr,
                                np.array(sim) - np.array(sim_sd),
                                np.array(sim) + np.array(sim_sd),
                                color=C_SIMVP, alpha=0.18,
                                label='SimVP ±1 std')
        ax.plot(lt_arr, unet, 's-', color=C_UNET, lw=2, label='Vanilla UNet (3 seed, ly)')
        ax.fill_between(lt_arr,
                        np.array(unet) - np.array(unet_sd),
                        np.array(unet) + np.array(unet_sd),
                        color=C_UNET, alpha=0.18, label='UNet ±1 std')
        if extra and all(v is not None for v in extra):
            ax.plot(lt_arr, extra, '--', color=C_PERSIST, lw=1.8, label='persistence')
        ax.set_xlabel('lead time (steps × 10 min)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if logy:
            ax.set_yscale('log')

    panel(axes[0], s_mse, s_mse_sd, u_mse, u_mse_sd, None,
          'MSE (normalized)', '(a) MSE vs lead time')
    panel(axes[1], s_rmse, s_rmse_sd, u_rmse, u_rmse_sd, persist,
          'RMSE (K)', '(b) RMSE in brightness temperature (K)')
    panel(axes[2], s_ssim, s_ssim_sd, u_ssim, u_ssim_sd, None,
          'SSIM', '(c) SSIM vs lead time')

    if n_seeds < 2:
        fig.suptitle('WARNING: only 1 SimVP seed available — no variance band '
                     '(run seeds 0/123 to enable)', color='red', fontsize=11)
    else:
        fig.suptitle('SimVP vs Vanilla UNet — autoregressive rollout '
                     f'(test set, 128 samples; SimVP {n_seeds} seeds)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    os.makedirs(osp.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f'[SAVE] 对比图 -> {args.out}')

    # 同时把画图用的数据落成 CSV，便于复算/重绘
    csv_path = osp.splitext(args.out)[0] + '.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lead_time', 'minutes',
                    'simvp_mse_mean', 'simvp_mse_std',
                    'simvp_rmse_K_mean', 'simvp_rmse_K_std',
                    'simvp_ssim_mean', 'simvp_ssim_std',
                    'unet_mse_mean', 'unet_mse_std',
                    'unet_rmse_K_mean', 'unet_rmse_K_std',
                    'unet_ssim_mean', 'unet_ssim_std',
                    'persistence_rmse_K'])
        for i, lt in enumerate(lts):
            w.writerow([lt, lt * 10,
                        s_mse[i], s_mse_sd[i], s_rmse[i], s_rmse_sd[i],
                        s_ssim[i], s_ssim_sd[i],
                        u_mse[i], u_mse_sd[i], u_rmse[i], u_rmse_sd[i],
                        u_ssim[i], u_ssim_sd[i], persist[i]])
    print(f'[SAVE] 数据 CSV -> {csv_path}')

    # 顺手打印 LT=18 的关键对照
    if s_mse[17] is not None:
        print('=' * 66)
        print(f'LT=18 (180 min): SimVP MSE={s_mse[17]:.4f}±{s_mse_sd[17]:.4f} | '
              f'UNet MSE={u_mse[17]:.4f}±{u_mse_sd[17]:.4f}')
        print(f'        RMSE(K): SimVP {s_rmse[17]:.2f} | UNet {u_rmse[17]:.2f} | '
              f'persistence {persist[17]:.2f}' if persist[17] else '')
        print('=' * 66)


if __name__ == '__main__':
    main()
