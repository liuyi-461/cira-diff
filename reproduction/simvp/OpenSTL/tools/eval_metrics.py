#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**8 指标评估套件**：BT PDF / PSD / SSIM / CRPS / 覆盖率 / 云团移动 / 冷却率 / 面积增长。

为什么单独写一个脚本
--------------------
`tools/eval_simvp_baseline.py` 只报 MAE/RMSE/SSIM/PSNR（逐像素口径）。要回答
"滚到第几步开始崩溃/变糊"，必须按**尺度**和**物理量**拆开误差：

| 指标 | 回答什么问题 |
|---|---|
| BT PDF | 输出是否过早向均值收缩（分布学证据） |
| PSD（径向平均） | **哪个空间尺度先塌**（预期先塌小尺度） |
| SSIM / CRPS | 结构相似度 / 概率口径（CRPS 对单成员退化为 MAE） |
| 覆盖率 + 面积增长 | 云体素准不准、生消节奏准不准 |
| 云团移动 | 位移误差（把"糊"和"位置偏"分开） |
| 冷却率 | 对流加深/减弱的趋势准不准 |
| LapVar 占真值比 | 高频能量（**必须与 PSD 同报**，单看它会奖励噪声） |

数据契约
--------
* `--preds`：`(18, N, 1, 256, 256)`，**zarr 归一化空间**（与 rollout 评估产物一致）；
  `--truth` 从 test zarr 的 `output_images` 取（同一归一化空间）。
* 冷芯阈值 **235 K**：与 `model_performance` / `dcm/wind_residual` 口径一致，便于横向比。
* 全部指标在**亮温 K** 空间计算（误差类先换算），PDF/PSD 用绝对亮温。
* 帧序列约定：`seq[0]` = 输入的最后一帧观测（t），`seq[k]` = 第 k 个预报帧（t+10k min）。
  因此"移动/冷却率/面积增长"这类**趋势量**在 k≥1 上有定义。

⚠️ 引用注意
-----------
* CRPS：本项目的模型是**确定性单成员**，CRPS ≡ MAE（解析退化）。仅当未来引入
  集合预报时才需按 zjt `datasets/goes/metrics.py` 的集合公式计算。
* 冷却率/面积增长中用于界定"冷芯区域"的掩膜取自**真值**，属于诊断型指标
  （回答"在该冷的地方模型有没有报对趋势"），**不能**当作业务评分。
* 云团移动用 DIS 光流；符号约定由合成位移自检确定（不靠记忆，见 `--selftest`）。

用法::

    python tools/eval_metrics.py \
        --preds work_dirs/satcast_simvp_full/baseline_eval_rollout18_full/preds.npy \
        --out-dir docs/experiments/EXP-008-metrics-baseline/baseline/ \
        --leads 1,3,6,9,12,15,18
"""

import argparse
import json
import os
import os.path as osp
import sys
import time

_HERE = osp.dirname(osp.abspath(__file__))
_OPENSTL_DIR = osp.dirname(_HERE)
for _p in (_OPENSTL_DIR, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import zarr  # noqa: E402

from openstl.datasets.dataloader_satcast import MEAN_K, STD_K  # noqa: E402

COLD_THRESH = 235.0                 # 冷芯阈值（K），与组内口径一致
PDF_BINS = np.arange(180.0, 321.0, 2.0)
HF_K = 40.0                         # 高频阈值：|k| > 40 即 <=6 像素尺度

# ---- 径向 PSD 的预计算网格（256x256） -------------------------------------
_N = 256
_freq = np.fft.fftfreq(_N) * _N
KX, KY = np.meshgrid(_freq, _freq)
_KR = np.sqrt(KX ** 2 + KY ** 2)
RAD_EDGES = np.arange(0.0, float(_N // 2) + 2.0, 1.0)     # 0 .. 130
_RAD_IDX = np.digitize(_KR.ravel(), RAD_EDGES[1:-1])
_RAD_CNT = np.bincount(_RAD_IDX, minlength=len(RAD_EDGES) - 1).astype(np.float64)
_KR_C = 0.5 * (RAD_EDGES[:-1] + RAD_EDGES[1:])
_HF_MASK = _KR > HF_K


def to_K(a):
    """归一化空间 -> 亮温 K。"""
    return np.asarray(a, dtype=np.float64) * STD_K + MEAN_K


# --------------------------------------------------------------- 指标函数 ---
def bt_pdf(frames):
    """(M,H,W) K -> 密度直方图（已归一到积分 1）。"""
    h, _ = np.histogram(np.asarray(frames, dtype=np.float64).ravel(),
                        bins=PDF_BINS, density=True)
    return h


def radial_psd(frame):
    """(H,W) K -> (径向波数, 径向平均功率谱)。先去均值，功率按像素数归一。"""
    f = np.asarray(frame, dtype=np.float64)
    f = f - f.mean()
    P = np.abs(np.fft.fft2(f)) ** 2 / f.size
    s = np.bincount(_RAD_IDX, weights=P.ravel(), minlength=len(_RAD_CNT))
    with np.errstate(invalid='ignore', divide='ignore'):
        prof = s / _RAD_CNT
    return _KR_C, np.nan_to_num(prof)


def hf_energy_frac(frame):
    """高频（|k|>40）能量占总能量比例。"""
    f = np.asarray(frame, dtype=np.float64)
    f = f - f.mean()
    P = np.abs(np.fft.fft2(f)) ** 2
    tot = P.sum()
    return float(P[_HF_MASK].sum() / tot) if tot > 0 else 0.0


def lapvar(frame):
    """Laplacian 方差（组内已发表表格同定义）。"""
    from scipy.ndimage import laplace
    return float(laplace(np.asarray(frame, dtype=np.float64)).var())


def cold_mask(frame_k):
    return np.asarray(frame_k) <= COLD_THRESH


# 组内 SSIM/CSI/FSS 的**逐位一致**实现。
# 来源：`zjt/cira-diff/flowcast_v2/datasets/goes/metrics.py::Metrics.update`
#   * 高斯滤波 σ=1.5（不是均值窗）
#   * c1=(0.01·data_range)²、c2=(0.03·data_range)²，data_range = KMAX−KMIN = 180.0078
#   * CSI：Tb ≤ 阈值的 hits/union；FSS：9×9 均值窗的 1−MSE/(2·power)
# 之所以照搬而不自作：本项目 §8.3 立过规矩——跨模型清晰度对比必须同一指标族。
_DR = 340.003758984375 - 159.99592773437502


def group_metrics_single(p, t, cold_thresh, fss_window=9):
    """单帧 (ssim, csi, fss)，口径与组内 Metrics 逐位一致。p/t 为亮温 K。"""
    from scipy.ndimage import uniform_filter, gaussian_filter
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    u = gaussian_filter(p, 1.5)
    v = gaussian_filter(t, 1.5)
    vp = np.maximum(gaussian_filter(p * p, 1.5) - u * u, 0)
    vt = np.maximum(gaussian_filter(t * t, 1.5) - v * v, 0)
    cov = gaussian_filter(p * t, 1.5) - u * v
    c1, c2 = (.01 * _DR) ** 2, (.03 * _DR) ** 2
    ssim = float(np.mean(((2 * u * v + c1) * (2 * cov + c2))
                         / ((u * u + v * v + c1) * (vp + vt + c2))))
    ep, et = p <= cold_thresh, t <= cold_thresh
    hits = float(np.logical_and(ep, et).sum())
    union = float(np.logical_or(ep, et).sum())
    csi = hits / union if union else None
    fp = uniform_filter(ep.astype(float), fss_window)
    ft = uniform_filter(et.astype(float), fss_window)
    den = float((fp * fp + ft * ft).sum())
    fss = 1.0 - float(np.square(fp - ft).sum()) / den if den else None
    return ssim, csi, fss


def flow_dis(a, b):
    """DIS medium 光流：把 a 推到 b 的**正向位移**。

    符号约定由 `--selftest` 的合成平移实验确定（把图平移 (+3,+2)，取还原误差小的符号），
    与 `dcm/wind_residual/pipeline/flowlib.py` 的做法一致——不靠记忆。
    """
    import cv2
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    a8 = _to_u8(a)
    b8 = _to_u8(b)
    fl = dis.calc(a8, b8, None)          # (H,W,2) = (dx, dy) 正向位移
    return fl


def _to_u8(frame_k):
    """K -> uint8（用固定物理范围，避免逐帧归一化把不同帧的对比度抹掉）。"""
    lo, hi = 160.0, 340.0
    x = (np.asarray(frame_k, dtype=np.float64) - lo) / (hi - lo)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def warp(img, flow, k):
    """按 k 倍位移外推（用于平流基线与自检）。"""
    from scipy.ndimage import map_coordinates
    h, w = img.shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    coords = np.stack([(gy - k * flow[..., 1]).ravel(),
                       (gx - k * flow[..., 0]).ravel()])
    return map_coordinates(img, coords, order=1, mode='nearest').reshape(h, w)


def selftest_flow_sign():
    """合成平移定标：返回 (误差_x_minus, 误差_x_plus, 选定符号)。"""
    rng = np.random.default_rng(0)
    img = rng.normal(280.0, 8.0, (256, 256))
    img = np.asarray(img, dtype=np.float64)
    dy, dx = 3.0, 2.0                      # 真值位移
    gy, gx = np.mgrid[0:256, 0:256].astype(np.float32)
    from scipy.ndimage import map_coordinates
    moved = map_coordinates(img, [gy + dy, gx + dx], order=1,
                            mode='nearest').reshape(256, 256)
    fl = flow_dis(img, moved)              # 期望 ≈ (dx, dy)
    err_minus = float(np.abs(warp(img, fl, 1) - moved).mean())
    err_plus = float(np.abs(warp(img, fl, -1) - moved).mean())
    sign = 'minus' if err_minus <= err_plus else 'plus'
    print(f'  [光流自检] 估计位移=({fl[..., 0].mean():.2f},{fl[..., 1].mean():.2f}) '
          f'真值=({dx},{dy}) | warp(-flow) MAE={err_minus:.4f} '
          f'warp(+flow) MAE={err_plus:.4f} -> 采用 {sign}')
    return err_minus, err_plus, sign


# ------------------------------------------------------------------- 主流程 ---
class UNetPreds:
    """把 `model_performance/unet/forecasts.npy` 适配成本脚本的预测接口。

    外部格式：`(N, 18, 1, H, W)`，**亮温 K**，float32，约 4.83 GB。
    本脚本内部约定：`(18, N, 1, H, W)`，**zarr 归一化空间**。

    这里用 mmap + 切片时转换，**不生成临时文件**（否则要 4.8 GB 磁盘）。
    转换包含两步：轴转置 + `K -> 归一化`（主循环随后会 to_K 转回来，
    往返一次只损失 float32 精度，约 1e-5 K，可忽略）。
    """

    def __init__(self, path):
        self.a = np.load(path, mmap_mode='r')
        if self.a.ndim != 5:
            raise ValueError(f'UNet forecasts 应为 5 维 (N,18,1,H,W)，'
                             f'实际 {self.a.shape}')
        self.shape = (self.a.shape[1], self.a.shape[0], self.a.shape[2],
                      self.a.shape[3], self.a.shape[4])

    def __getitem__(self, idx):
        lead, samples, _ch = idx
        k = np.asarray(self.a[samples, :, 0], dtype=np.float32)   # (b,18,H,W) K
        k = np.transpose(k, (1, 0, 2, 3))                        # (18,b,H,W) K
        return (k - MEAN_K) / STD_K                              # -> 归一化


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default=None,
                    help='(18,N,1,256,256) 归一化空间的滚动预报；'
                         '用 --unet / --persistence 时可省略')
    ap.add_argument('--unet', default=None,
                    help='直接读 model_performance/unet/forecasts.npy'
                         '（(N,18,1,H,W) 亮温 K），自动转轴 + 归一化，'
                         '免去生成 4.8 GB 临时文件')
    ap.add_argument('--truth-zarr', '--data', dest='data',
                    default=os.environ.get(
                        'SATCAST_ROOT',
                        '/home/group1/26fall_aiclass/yr/data1/data1/satcast')
                    + '/edm_GOES_ch13_test_dataset.zarr')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--leads', default='1,3,6,9,12,15,18',
                    help='要做趋势量(移动/冷却/面积)的时效；PDF/PSD 等总是全 18 时效')
    ap.add_argument('--max-samples', type=int, default=None,
                    help='默认全量 1024')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--cold-thresh', type=float, default=COLD_THRESH)
    ap.add_argument('--selftest', action='store_true',
                    help='只跑光流符号自检后退出')
    ap.add_argument('--persistence', action='store_true',
                    help='不读 --preds，直接用「最后一帧观测」作预测（口径锚点校验用）')
    args = ap.parse_args()

    if args.selftest:
        selftest_flow_sign()
        return

    leads = [int(x) for x in args.leads.split(',') if x.strip()]
    if args.unet:
        P = UNetPreds(args.unet)
        print(f'[MODE] UNet（official，model_performance）：'
              f'轴 (N,18,1,H,W) 亮温K -> (18,N,1,H,W) 归一化空间，共 {P.shape[1]} 条')
    elif args.persistence:
        # 伪 preds：直接用输入最后一帧，形状与真实 preds 一致但懒生成
        P = None
        print('[MODE] persistence：预测 = 输入最后一帧（锚点校验）')
    else:
        if not args.preds:
            raise SystemExit('必须给 --preds、--unet 或 --persistence 之一')
        P = np.load(args.preds, mmap_mode='r')            # (18,N,1,H,W)
        assert P.ndim == 5, f'--preds 应为 (18,N,1,H,W)，实际 {P.shape}'
    n_lead, n_all = ((P.shape[0], P.shape[1]) if P is not None else (18, 1024))
    n = n_all if args.max_samples is None else min(n_all, args.max_samples)
    g = zarr.open(args.data, mode='r')
    Y = g['output_images']                            # (N,18,H,W) 归一化
    X = g['input_images']                             # (N,2,H,W)  归一化
    assert Y.shape[0] >= n and X.shape[0] >= n, '真值/输入样本数不足'

    # ---------- 累加器 ----------
    pdf_p = np.zeros(len(PDF_BINS) - 1)
    pdf_t = np.zeros(len(PDF_BINS) - 1)
    psd_p = np.zeros(len(_KR_C))
    psd_t = np.zeros(len(_KR_C))
    # 逐 lead 累加：LapVar 与高频能量占比是回答"第几步开始糊"的核心指标
    hf_p = np.zeros(n_lead)
    hf_t = np.zeros(n_lead)
    lap_p = np.zeros(n_lead)
    lap_t = np.zeros(n_lead)
    cov_p = np.zeros(n_lead)
    cov_t = np.zeros(n_lead)
    cov_cnt = 0
    ssim_sum = np.zeros(n_lead)
    csi_hits = np.zeros(n_lead)
    csi_union = np.zeros(n_lead)
    fss_err = np.zeros(n_lead)
    fss_ref = np.zeros(n_lead)
    mae_sum = np.zeros(n_lead)
    rmse_sum = np.zeros(n_lead)
    frame_cnt = 0
    # 趋势量（只在 leads 上算）
    mot_err = {k: [] for k in leads}       # 位移矢量误差（像素/10min）
    cool_err = {k: [] for k in leads}      # 冷芯平均 BT 倾向误差（K/10min）
    area_err = {k: [] for k in leads}      # 冷芯面积误差（像素）
    growth_err = {k: [] for k in leads}    # 面积增长误差（像素/10min）

    t0 = time.time()
    for s0 in range(0, n, args.batch):
        s1 = min(s0 + args.batch, n)
        b = s1 - s0
        # (b, lead, H, W) K —— 注意 preds 的轴序是 (lead, N, C, H, W)，必须转置，
        # 否则 pr[j] 会取到"第 j 个 lead 的全部样本"而不是"第 j 个样本的全部 lead"
        # （该错误曾让 p_seq 变成 b+1 帧而越界；已加断言防再犯）。
        obs_last = to_K(np.asarray(X[s0:s1, 1], dtype=np.float32))   # (b,H,W)
        # persistence 模式：预测 = 最后一帧观测重复 n_lead 次（锚点校验用）
        pr = (to_K(np.asarray(P[:, s0:s1, 0], dtype=np.float32)).transpose(1, 0, 2, 3)
              if P is not None else np.repeat(obs_last[:, None], n_lead, axis=1))
        tr = to_K(np.asarray(Y[s0:s1], dtype=np.float32))
        assert pr.shape[1] == n_lead == tr.shape[1], \
            f'轴序错误：pr={pr.shape} tr={tr.shape}，期望第二维均为 lead={n_lead}'
        assert pr.shape[0] == tr.shape[0] == obs_last.shape[0] == b, \
            f'批大小不一致 pr={pr.shape} tr={tr.shape} obs={obs_last.shape}'

        for j in range(b):
            p_seq = np.concatenate([obs_last[j][None], pr[j]], axis=0)   # (19,H,W)
            t_seq = np.concatenate([obs_last[j][None], tr[j]], axis=0)
            for k in range(n_lead):
                pf, tf = p_seq[k + 1], t_seq[k + 1]
                pdf_p += bt_pdf(pf)
                pdf_t += bt_pdf(tf)
                _, pp = radial_psd(pf)
                _, tp = radial_psd(tf)
                psd_p += pp
                psd_t += tp
                hf_p[k] += hf_energy_frac(pf)
                hf_t[k] += hf_energy_frac(tf)
                lap_p[k] += lapvar(pf)
                lap_t[k] += lapvar(tf)
                mp = cold_mask(pf)
                mt = cold_mask(tf)
                cov_p[k] += mp.mean()
                cov_t[k] += mt.mean()
                d = pf - tf
                mae_sum[k] += float(np.abs(d).mean())
                rmse_sum[k] += float((d ** 2).mean())
                # SSIM / CSI / FSS：口径与组内 Metrics 逐位一致（见 group_metrics_single）
                s, _, _ = group_metrics_single(pf, tf, args.cold_thresh)
                ssim_sum[k] += s
                # CSI/FSS 用累加器口径（hits/union 与 1−MSE/power），与组内一致
                ep, et = pf <= args.cold_thresh, tf <= args.cold_thresh
                csi_hits[k] += float(np.logical_and(ep, et).sum())
                csi_union[k] += float(np.logical_or(ep, et).sum())
                from scipy.ndimage import uniform_filter
                fpw = uniform_filter(ep.astype(float), 9)
                ftw = uniform_filter(et.astype(float), 9)
                fss_err[k] += float(np.square(fpw - ftw).sum())
                fss_ref[k] += float((fpw * fpw + ftw * ftw).sum())
            frame_cnt += 1
            # 趋势量（只在指定 leads 上算，节省光流开销）
            # 索引约定：seq[0] = 最后一帧观测，seq[k] = 第 k 个预报帧（1-based lead）
            for k in leads:
                if k < 2 or k > n_lead:
                    continue          # 趋势量需要 k 与 k-1 两帧，故从 LT=2 起
                mt = cold_mask(t_seq[k])
                mt_prev = cold_mask(t_seq[k - 1])
                # 云团移动：真值/预报各自在 (k-1 -> k) 的位移场之差
                ft = flow_dis(t_seq[k - 1], t_seq[k])
                fp = flow_dis(p_seq[k - 1], p_seq[k])
                mot_err[k].append(float(np.abs(fp - ft).mean()))
                # 冷却率：在"真值冷芯"区域上的平均 BT 倾向误差（诊断型，见 docstring）
                region = mt | mt_prev
                if region.any():
                    cool_err[k].append(float(
                        (p_seq[k][region].mean() - p_seq[k - 1][region].mean())
                        - (t_seq[k][region].mean() - t_seq[k - 1][region].mean())))
                # 冷芯面积误差（预报 − 真值）
                mp_k = cold_mask(p_seq[k])
                area_err[k].append(int(mp_k.sum() - mt.sum()))
                # 面积增长误差：(k 的面积 − k-1 的面积) 之差
                gp_prev = cold_mask(p_seq[k - 1])
                tp_prev = cold_mask(t_seq[k - 1])
                growth_err[k].append(float(
                    (mp_k.sum() - gp_prev.sum()) - (mt.sum() - tp_prev.sum())))
        if (s0 // args.batch) % 8 == 0:
            el = time.time() - t0
            print(f'  {s1}/{n}  {el:.0f}s  预计 {el * n / max(s1,1):.0f}s', flush=True)

    # ---------- 汇总 ----------
    def _mean(x):
        return float(np.mean(x)) if len(x) else None

    nrm = max(frame_cnt, 1)
    # 预测来源的可读标注：三种模式（--unet / --persistence / --preds）都要能写清
    if args.unet:
        _src = f'UNet(official, model_performance): {osp.abspath(args.unet)}'
    elif args.persistence:
        _src = 'PERSISTENCE(输入最后一帧)'
    else:
        _src = osp.abspath(args.preds)
    out = {
        'meta': {
            'preds': _src,
            'truth_zarr': osp.abspath(args.data),
            'n_samples': n, 'n_lead': n_lead,
            'cold_thresh_K': args.cold_thresh,
            'ssim_csi_fss_convention': 'zjt flowcast_v2/datasets/goes/metrics.py '
                                       '（gaussian σ=1.5, data_range=180.008, '
                                       'FSS 窗 9, CSI 阈值=Tb≤%.0fK）' % args.cold_thresh,
            'pdf_bins_K': PDF_BINS.tolist(),
            'leads_trend': leads,
            'crps_note': '单成员确定性模型：CRPS 解析退化为 MAE',
            'elapsed_s': round(time.time() - t0, 1),
        },
        # 池化口径（全部 18 时效合计）——与 model_performance / dcm 的总表可比
        'pooled': {
            'mae_K': float(np.sum(mae_sum)) / (frame_cnt * n_lead),
            'mse_K2': float(np.sum(rmse_sum)) / (frame_cnt * n_lead),
            'rmse_K': float(np.sqrt(np.sum(rmse_sum) / (frame_cnt * n_lead))),
            'crps_K': float(np.sum(mae_sum)) / (frame_cnt * n_lead),
            'ssim': float(np.sum(ssim_sum)) / (frame_cnt * n_lead),
            'csi': (float(np.sum(csi_hits) / np.sum(csi_union))
                    if np.sum(csi_union) > 0 else None),
            'fss': (1.0 - float(np.sum(fss_err) / np.sum(fss_ref))
                    if np.sum(fss_ref) > 0 else None),
            'hf_energy_frac_pred': float(np.mean(hf_p)),
            'hf_energy_frac_truth': float(np.mean(hf_t)),
            'lapvar_pred': float(np.mean(lap_p)),
            'lapvar_truth': float(np.mean(lap_t)),
            'lapvar_frac_of_truth': float(np.sum(lap_p) / max(np.sum(lap_t), 1e-12)),
            'coverage_pred': float(np.sum(cov_p)) / (frame_cnt * n_lead),
            'coverage_truth': float(np.sum(cov_t)) / (frame_cnt * n_lead),
        },
        'bt_pdf_density': {'pred': (pdf_p / frame_cnt).tolist(),
                           'truth': (pdf_t / frame_cnt).tolist()},
        'psd_radial': {'k': _KR_C.tolist(),
                       'pred': (psd_p / frame_cnt).tolist(),
                       'truth': (psd_t / frame_cnt).tolist()},
        'per_lead': {},
        'trend': {},
    }
    for k in range(n_lead):
        out['per_lead'][str(k + 1)] = {
            # 注意：累加时已是"逐样本整图均值"，故汇总只除样本数，不再除像素数
            # （曾误除 256*256 导致 RMSE 显示 0.023 而非 ~5 K）
            'mae_K': mae_sum[k] / max(frame_cnt, 1),
            'rmse_K': float(np.sqrt(rmse_sum[k] / max(frame_cnt, 1))),
            'crps_K': mae_sum[k] / max(frame_cnt, 1),   # ≡ MAE（见 meta.crps_note）
            'ssim': ssim_sum[k] / max(frame_cnt, 1),
            'csi': (csi_hits[k] / csi_union[k]) if csi_union[k] > 0 else None,
            'fss': (1.0 - fss_err[k] / fss_ref[k]) if fss_ref[k] > 0 else None,
            'coverage_pred': cov_p[k] / max(frame_cnt, 1),
            'coverage_truth': cov_t[k] / max(frame_cnt, 1),
            'coverage_diff': (cov_p[k] - cov_t[k]) / max(frame_cnt, 1),
            'hf_energy_frac_pred': hf_p[k] / max(frame_cnt, 1),
            'hf_energy_frac_truth': hf_t[k] / max(frame_cnt, 1),
            'lapvar_pred': lap_p[k] / max(frame_cnt, 1),
            'lapvar_truth': lap_t[k] / max(frame_cnt, 1),
            'lapvar_frac_of_truth': float(
                lap_p[k] / max(lap_t[k], 1e-12)),
            'sharpness_ratio': None,
        }
    # 高频/锐度是"每帧均值"，不是每 lead 的（这里按 lead 也算了，单独填回）
    # —— 上面的循环已按 lead 累加，下面补算每 lead 的标量
    # 由于 hf/lap 是按帧累加成标量，这里改为逐 lead 重算代价高，
    # 因此只报告全时效均值，并保留每 lead 的曲线占位为 None。
    # 兼容旧键名：'overall' 现由 'pooled' 承担，此处仅转发，避免下游读不到
    out['overall'] = dict(out['pooled'])
    out['overall']['n_frames'] = frame_cnt * n_lead
    for k in leads:
        out['trend'][str(k)] = {
            'cloud_motion_err_px': _mean(mot_err[k]),
            'cooling_rate_err_K': _mean(cool_err[k]),
            'cold_area_err_px': _mean(area_err[k]),
            'area_growth_err_px': _mean(growth_err[k]),
        }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(osp.join(args.out_dir, 'metrics_8.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    _print_table(out)
    _plot(out, args.out_dir)
    _print_verdict(out)
    print(f'\n[SAVE] {osp.join(args.out_dir, "metrics_8.json")}')


def _print_table(out):
    print('\n' + '-' * 84)
    print(f'{"LT":>3} {"RMSE(K)":>9} {"MAE(K)":>8} {"SSIM":>7} {"CRPS=MAE":>9} '
          f'{"CSI":>7} {"FSS":>7} {"覆盖pred":>8} {"差":>7}')
    print('-' * 84)
    for k in range(int(out['meta']['n_lead'])):
        d = out['per_lead'][str(k + 1)]
        csi = f'{d["csi"]:7.4f}' if d['csi'] is not None else '      —'
        fss = f'{d["fss"]:7.4f}' if d['fss'] is not None else '      —'
        print(f'{k+1:>3} {d["rmse_K"]:>9.3f} {d["mae_K"]:>8.3f} {d["ssim"]:>7.4f} '
              f'{d["crps_K"]:>9.3f} {csi} {fss} {d["coverage_pred"]:>8.3f} '
              f'{d["coverage_diff"]:>+7.3f}')
    o = out['pooled']
    print('-' * 84)
    print(f'[池化 18 时效，可与 model_performance/dcm 总表直接比]')
    print(f'  MAE={o["mae_K"]:.4f} K  RMSE={o["rmse_K"]:.4f} K  '
          f'CRPS(=MAE)={o["crps_K"]:.4f} K')
    print(f'  SSIM={o["ssim"]:.4f}  CSI={o["csi"] if o["csi"] is None else round(o["csi"],4)}'
          f'  FSS={o["fss"] if o["fss"] is None else round(o["fss"],4)}')
    print(f'  高频能量占比: pred={o["hf_energy_frac_pred"]:.4f} '
          f'truth={o["hf_energy_frac_truth"]:.4f}')
    print(f'  LapVar: pred={o["lapvar_pred"]:.3f} truth={o["lapvar_truth"]:.3f} '
          f'占真值 {o["lapvar_frac_of_truth"]*100:.1f}%'
          f'（>100% 可能已是噪声，须与 PSD 同判）')
    print('趋势量（lead: 移动误差px | 冷却率误差K | 面积误差px | 增长误差px）:')
    for k, v in out['trend'].items():
        # ⚠️ trend 对 k<2 无定义（需要 k 与 k-1 两帧），_mean 会返回 None，
        # 直接 :.3f 格式化会抛 TypeError。这里统一降级为 "—"。
        def _f(x, fmt='%.3f'):
            return fmt % x if x is not None else '—'
        print(f'  LT={k:>2}: {_f(v["cloud_motion_err_px"])} | '
              f'{_f(v["cooling_rate_err_K"], "%+.3f")} | '
              f'{_f(v["cold_area_err_px"], "%+.0f")} | '
              f'{_f(v["area_growth_err_px"], "%+.0f")}')


def _plot(out, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_style import use_cjk_font
    use_cjk_font()
    os.makedirs(out_dir, exist_ok=True)
    n_lead = int(out['meta']['n_lead'])
    lts = np.arange(1, n_lead + 1)

    # 图 1：BT PDF
    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    c = 0.5 * (np.array(out['meta']['pdf_bins_K'][:-1])
               + np.array(out['meta']['pdf_bins_K'][1:]))
    ax.plot(c, out['bt_pdf_density']['truth'], '-', color='#1f77b4', lw=2,
            label='真值 truth')
    ax.plot(c, out['bt_pdf_density']['pred'], '--', color='#d62728', lw=2,
            label='预报 forecast')
    ax.set_xlabel('亮度温度 (K)', fontsize=11)
    ax.set_ylabel('概率密度', fontsize=11)
    ax.set_title('BT 概率密度函数（全时效合计）', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.savefig(osp.join(out_dir, 'bt_pdf.png'), dpi=140)
    plt.close(fig)

    # 图 2：径向平均 PSD
    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    k = np.array(out['psd_radial']['k'])
    ax.loglog(k[1:], np.array(out['psd_radial']['truth'])[1:], '-',
              color='#1f77b4', lw=2, label='真值 truth')
    ax.loglog(k[1:], np.array(out['psd_radial']['pred'])[1:], '--',
              color='#d62728', lw=2, label='预报 forecast')
    ax.axvline(HF_K, color='#888888', ls=':', lw=1.2,
               label=f'高频阈值 |k|>{HF_K:.0f}')
    ax.set_xlabel('径向波数 |k|（像素⁻¹）', fontsize=11)
    ax.set_ylabel('功率谱密度', fontsize=11)
    ax.set_title('径向平均功率谱：看哪个尺度先塌', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which='both')
    fig.savefig(osp.join(out_dir, 'psd_radial.png'), dpi=140)
    plt.close(fig)

    # 图 3：覆盖率 / 面积增长 / 冷却率 / 移动误差
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    cp = [out['per_lead'][str(i)]['coverage_pred'] for i in lts]
    ct = [out['per_lead'][str(i)]['coverage_truth'] for i in lts]
    axes[0, 0].plot(lts, ct, 'o-', color='#1f77b4', lw=2, label='真值')
    axes[0, 0].plot(lts, cp, 's--', color='#d62728', lw=2, label='预报')
    axes[0, 0].set_title(f'冷芯覆盖率（Tb ≤ {out["meta"]["cold_thresh_K"]:.0f} K）',
                         fontsize=11)
    axes[0, 0].set_xlabel('时效（步）')
    axes[0, 0].set_ylabel('覆盖率')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)

    tk = sorted(int(k) for k in out['trend'])
    axes[0, 1].plot(tk, [out['trend'][str(k)]['cloud_motion_err_px'] for k in tk],
                    'D-', color='#ff7f0e', lw=2)
    axes[0, 1].set_title('云团移动误差（位移矢量差）', fontsize=11)
    axes[0, 1].set_xlabel('时效（步）')
    axes[0, 1].set_ylabel('像素 / 10 min')
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(tk, [out['trend'][str(k)]['cooling_rate_err_K'] for k in tk],
                    '^-', color='#2ca02c', lw=2)
    axes[1, 0].axhline(0, color='#888888', lw=1)
    axes[1, 0].set_title('冷芯冷却率误差（预报 − 真值）', fontsize=11)
    axes[1, 0].set_xlabel('时效（步）')
    axes[1, 0].set_ylabel('K / 10 min')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(tk, [out['trend'][str(k)]['area_growth_err_px'] for k in tk],
                    'v-', color='#9467bd', lw=2)
    axes[1, 1].axhline(0, color='#888888', lw=1)
    axes[1, 1].set_title('冷芯面积增长误差（预报 − 真值）', fontsize=11)
    axes[1, 1].set_xlabel('时效（步）')
    axes[1, 1].set_ylabel('像素 / 10 min')
    axes[1, 1].grid(alpha=0.3)
    fig.savefig(osp.join(out_dir, 'coverage_trend.png'), dpi=140)
    plt.close(fig)
    print(f'  [SAVE] bt_pdf.png / psd_radial.png / coverage_trend.png')


def _print_verdict(out):
    """自动判读：回答"滚到第几步还稳定、从哪步开始崩溃"。

    四条客观判据（**规则预设，不替代人工判读**）：
      A. 锐度崩塌点：逐 lead LapVar 占真值比例首次低于 50% / 30% 的时效
      B. 尺度崩塌点：逐 lead 高频能量占比（预报/真值）首次低于 50% 的时效
      C. 位置/生偏点：逐 lead 云团移动误差首次超过 0.5 像素/10min 的时效
      D. 稳定性：RMSE 增速首次降到 <5%/步（此后进入"线性变差"而非"骤崩"）
    """
    n_lead = int(out['meta']['n_lead'])
    pl = out['per_lead']
    o = out['pooled']
    print('\n' + '=' * 74)
    print('自动判读：滚到第几步还稳定？从哪步开始崩溃？（规则预设，需人工复核）')
    print('=' * 74)

    frac = [pl[str(i + 1)]['lapvar_frac_of_truth'] for i in range(n_lead)]
    hf_r = [pl[str(i + 1)]['hf_energy_frac_pred']
            / max(pl[str(i + 1)]['hf_energy_frac_truth'], 1e-12)
            for i in range(n_lead)]
    r = [pl[str(i + 1)]['rmse_K'] for i in range(n_lead)]
    dr = [(r[i] / r[max(i - 1, 0)] - 1) * 100 if i > 0 else 0.0
          for i in range(n_lead)]

    a50 = next((i + 1 for i, f in enumerate(frac) if f < 0.50), None)
    a30 = next((i + 1 for i, f in enumerate(frac) if f < 0.30), None)
    b50 = next((i + 1 for i, f in enumerate(hf_r) if f < 0.50), None)
    d_slow = next((i + 1 for i in range(1, n_lead) if dr[i] < 5.0), None)

    print(f'  A. 锐度崩塌：LapVar 占真值比例 LT=1 为 {frac[0]*100:.0f}%，'
          f'LT={n_lead} 为 {frac[-1]*100:.0f}%')
    print(f'     → 首次 <50% 于 LT={a50}；首次 <30% 于 LT={a30}'
          f'（<30% 可认为"只剩大尺度形态"）')
    print(f'  B. 尺度崩塌：高频能量占比（预报/真值）LT=1 为 {hf_r[0]:.2f}，'
          f'LT={n_lead} 为 {hf_r[-1]:.2f}')
    print(f'     → 首次 <0.5 于 LT={b50}（小尺度先塌的直接证据）')
    print(f'  D. 稳定性：RMSE 增速首次降到 <5%/步于 LT={d_slow}'
          f'（此后为"线性变差"而非"骤崩"）')

    mot = {int(k): v['cloud_motion_err_px']
           for k, v in out['trend'].items()
           if v['cloud_motion_err_px'] is not None}
    if mot:
        ks = sorted(mot)
        c05 = next((k for k in ks if mot[k] > 0.5), None)
        print(f'  C. 位置/位移：云团移动误差 LT={ks[0]} 为 {mot[ks[0]]:.3f} px/10min，'
              f'LT={ks[-1]} 为 {mot[ks[-1]]:.3f} px/10min')
        print(f'     → 首次超过 0.5 px/10min 于 LT={c05}')

    covd = [pl[str(i + 1)]['coverage_diff'] for i in range(n_lead)]
    print(f'  覆盖率偏差: LT=1 {covd[0]:+.3f} -> LT={n_lead} {covd[-1]:+.3f}'
          f'（负=云量报少了，即"生消"偏差）')
    print(f'  池化 LapVar 占真值 = {o["lapvar_frac_of_truth"]*100:.1f}%'
          f'（>100% 可能已是噪声，必须与 PSD 同判）')
    print('=' * 74)
    print('  ⚠️ 以上为规则化初筛。最终"稳定可信预报时长"须由人工结合 '
          'BT PDF / PSD 曲线与单样本云图判定。')


if __name__ == '__main__':
    main()
