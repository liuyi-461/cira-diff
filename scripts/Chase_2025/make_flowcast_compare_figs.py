"""FlowCast / EDM / UNet 三模型逐样本对比图 + 锐度汇总（点 ▶ 直接跑，无需命令行参数）。

要改的地方都在「改这里」区：
    SAMPLE    画第几个样本（0~1023）
    LEADS     画哪些预报时刻（0=10min, 1=20min, ... 17=180min）
    COLUMNS   每一列画什么（加 CorrDiff 就往里加一行）
    CMAP      色标；Spectral_r = 高温偏红、低温偏蓝
    VMIN/VMAX 颜色温度范围 (K)

依赖：numpy / scipy / zarr / matplotlib（base python 即可）。
"""
import json
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import laplace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os

# 根目录可用环境变量覆盖；默认值是共享旧机 test-ai 上的位置
SATCAST = os.environ.get("SATCAST_DIR", "/data1/satcast")
DCM = os.environ.get("DCM_ROOT", "/home/group1/26fall_aiclass/dcm")

ROOT = Path(__file__).resolve().parents[1]                       # model_performance/ 根目录
DATA = f"{SATCAST}/edm_GOES_ch13_test_dataset.zarr"          # 原始数据（取 truth / input）
MEAN, STD = 279.0699, 19.3297                                    # 归一化：K = x * STD + MEAN

# ===================== 改这里 =====================
SAMPLE = 128                                      # 画第几个 case（0~1023）
LEADS = [0, 5, 11, 17]                            # 10min / 1h / 2h / 3h 四个代表时刻
LEAD_NAMES = ["10min", "1h", "2h", "3h"]          # 与 LEADS 一一对应的纵轴标签
CMAP = "Spectral_r"                               # 统一色标：高温偏红、低温偏蓝
VMIN, VMAX = 200.0, 300.0                         # 颜色温度范围 (K)
OUT_DIR = ROOT / "figs"
# ===================================================

# ---- 载入该样本（mmap 只读所需切片，不加载整个大文件） ----------------------
fc = np.load(ROOT / "flowcast/forecasts.npy", mmap_mode="r")[SAMPLE]    # (18,8,H,W)
edm = np.load(ROOT / "edm/forecasts.npy", mmap_mode="r")[SAMPLE]        # (18,8,H,W)
unet = np.load(ROOT / "unet/forecasts.npy", mmap_mode="r")[SAMPLE]      # (18,1,H,W)
vae = np.load(ROOT / "vae_recon/vae_recon.npy", mmap_mode="r")[SAMPLE]  # (18,H,W) VAE 重建
raw = zarr.open(DATA, mode="r")
truth = raw["output_images"][SAMPLE] * STD + MEAN      # (18,H,W) K
inp = raw["input_images"][SAMPLE] * STD + MEAN         # (2,H,W) K

fc_mean = fc.mean(1)      # 集合均值 (18,H,W)
edm_mean = edm.mean(1)
unet = unet[:, 0]         # 单成员

# ---- 每一列画什么（加 CorrDiff：先 np.load(mmap)，再加一行） ----------------
COLUMNS = [
    ("last obs",  lambda lead: inp[-1]),
    ("truth",     lambda lead: truth[lead]),
    ("VAE recon", lambda lead: vae[lead]),
    ("FlowCast",  lambda lead: fc_mean[lead]),
    ("EDM",       lambda lead: edm_mean[lead]),
    ("UNet",      lambda lead: unet[lead]),
]


def sharpness(img):
    """(Laplacian 方差, 平均梯度幅值)；越大越锐。"""
    f = np.asarray(img, dtype=np.float64)
    lap = laplace(f)
    gy, gx = np.gradient(f)
    return float(lap.var()), float(np.sqrt(gx ** 2 + gy ** 2).mean())


# ---- 画对比图 -----------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
n = len(COLUMNS)
fig, axes = plt.subplots(len(LEADS), n, figsize=(n * 2.2, len(LEADS) * 2.2),
                         squeeze=False, constrained_layout=True)
for r, lead in enumerate(LEADS):
    for c, (name, get) in enumerate(COLUMNS):
        ax = axes[r, c]
        img = get(lead)
        im = ax.imshow(img, cmap=CMAP, vmin=VMIN, vmax=VMAX)
        ax.set_title(f"{name}\nLapVar={sharpness(img)[0]:.2e}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[r, 0].set_ylabel(LEAD_NAMES[r], fontsize=9)
fig.suptitle(f"case {SAMPLE}", fontsize=11)
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02,
             label="brightness temperature (K)")
fig_path = OUT_DIR / f"sample_{SAMPLE:03d}.png"
fig.savefig(fig_path, dpi=110)
print(f"wrote {fig_path}")

# ---- 锐度汇总（参考） -----------------------------------------------------------
SHARPNESS = {
    "last obs":         lambda lead: inp[-1],
    "truth":            lambda lead: truth[lead],
    "VAE recon":        lambda lead: vae[lead],
    "FlowCast_mean":    lambda lead: fc_mean[lead],
    "FlowCast_member0": lambda lead: fc[lead, 0],
    "EDM_mean":         lambda lead: edm_mean[lead],
    "EDM_member0":      lambda lead: edm[lead, 0],
    "UNet":             lambda lead: unet[lead],
}
summary = {}
for name, get in SHARPNESS.items():
    lap = grad = 0.0
    for lead in LEADS:
        l, g = sharpness(get(lead))
        lap += l; grad += g
    m = len(LEADS)
    summary[name] = dict(lap_var=lap / m, grad_mag=grad / m)

print(f"\n锐度汇总（case {SAMPLE}，{len(LEADS)} 个 lead 平均；lap_var 越大越锐）：")
for name, v in summary.items():
    print(f"  {name:18s} lap_var={v['lap_var']:.4f}  grad_mag={v['grad_mag']:.4f}")
(OUT_DIR / "sharpness.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

# 锐度柱状图
fig, ax = plt.subplots(figsize=(8, 4))
names = list(summary)
ax.bar(names, [summary[k]["lap_var"] for k in names])
ax.set_ylabel("variance of Laplacian (sharpness)")
ax.set_title(f"sharpness, case {SAMPLE} (higher = sharper)")
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(OUT_DIR / "sharpness.png", dpi=120)
print(f"wrote {OUT_DIR / 'sharpness.png'}")
