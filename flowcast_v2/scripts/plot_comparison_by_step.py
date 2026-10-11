"""Plot the 18-step autoregressive forecast against the raw test target, every 3 frames.

直接用 VS Code 点击运行：只改下面的 INDEX（测试集样本序号），其余保持默认即可。
输出：runs/comparison_by_step/index_XXXX.png（2 行 × 6 列，每 30 分钟一例）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # flowcast_v2/
sys.path.insert(0, str(ROOT))                       # 保证 VS Code 直接运行时也能 import 项目模块
import numpy as np
import torch
import zarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets.goes.normalization import sample_count, to_kelvin
from datasets.goes.runtime import load_vae, load_flowcast
from scripts.evaluate import forecast, decode

# ============================ 只改这一段 ============================
INDEX = 63              # 测试集样本序号，取值 0 ~ 1023（运行时屏幕会打印实际范围）
SAMPLES = 8               # 集合成员数（图上显示其平均）
STEPS = 10                # ODE 采样步数（与训练一致）
EVERY = 3                 # 每隔几帧取一个面板：18/3 = 6 例
FRAME_MIN = 10            # 数据时间间隔（分钟）：GOES-16 全盘 10 min
CMAP = 'Greys'            # 灰度色标（冷云 = 浅色）
VMIN, VMAX = 200.0, 320.0  # 色标范围（K）；想要更大对比度就把区间收窄
DPI = 150
# ===================================================================

DATA = Path('/home/group1/26fall_aiclass/zjt/cira-diff/datasets/satcast')
PATHS = {
    'data':     DATA / 'edm_GOES_ch13_test_dataset.zarr',
    'latent':   ROOT / 'runs/latent/test.zarr',
    'vae_ckpt': ROOT / 'runs/vae/vae_1005_150523/models/early_stopping_model.pt',
    'vae_cfg':  ROOT / 'runs/vae/vae_1005_150523/config_snapshot.yaml',
    'cfm_ckpt': ROOT / 'runs/cfm/cfm_1007_010810/models/early_stopping_model.pt',
    'cfm_cfg':  ROOT / 'runs/cfm/cfm_1007_010810/config_snapshot.yaml',
}
OUT_DIR = ROOT / 'runs/comparison_by_step'


def setup_fonts():
    """Times New Roman；服务器没装则退回等宽度克隆 Nimbus Roman / Liberation Serif。"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif'],
        'font.size': 18,
        'axes.titlesize': 20,
        'figure.titlesize': 24,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
    })


def main():
    setup_fonts()
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw = zarr.open(str(PATHS['data']), mode='r')
    n = sample_count(raw)
    print(f'[info] 测试集共 {n} 条样本，INDEX 可用范围 0 ~ {n - 1}')
    if not 0 <= INDEX < n:
        raise SystemExit(f'INDEX={INDEX} 超出范围，请改成 0 ~ {n - 1}')

    vae, norm = load_vae(str(PATHS['vae_ckpt']), str(PATHS['vae_cfg']), device)
    latent = zarr.open(str(PATHS['latent']), mode='r')

    def shape(key):
        _, t, c, h, w = latent[key].shape
        return (t, h, w, c)

    model = load_flowcast(str(PATHS['cfm_ckpt']), str(PATHS['cfm_cfg']),
                          shape('input_images'), shape('output_images'), device)

    y = np.asarray(raw['output_images'][INDEX:INDEX + 1], dtype=np.float32)
    truth = to_kelvin(y, norm)[0]                                    # (T, 256, 256) 开尔文
    horizon = truth.shape[0] // shape('output_images')[0]            # 自回归步数（=18）

    cond = torch.from_numpy(np.asarray(latent['input_images'][INDEX:INDEX + 1]))
    cond = cond.permute(0, 1, 3, 4, 2).to(device)                    # (B, lag, H, W, C)
    target_shape = (1, *shape('output_images'))

    print(f'[info] 自回归 {horizon} 步 × {SAMPLES} 个集合成员 × {STEPS} 步 ODE，请稍候……')
    with torch.inference_mode():
        pred = np.stack([decode(vae, z, norm).cpu().numpy()
                         for z in forecast(model, cond, target_shape, SAMPLES, STEPS, horizon)])
    mean = pred.mean(0)[0][:horizon]                                 # 集合均值 (T, 256, 256)

    steps = np.arange(EVERY - 1, horizon, EVERY)                     # 2, 5, 8, 11, 14, 17
    fig, axes = plt.subplots(2, len(steps), figsize=(3.0 * len(steps), 7.2), constrained_layout=True)
    for j, k in enumerate(steps):
        for row, (field, name) in enumerate(((truth, 'Truth'), (mean, 'Forecast'))):
            ax = axes[row, j]
            im = ax.imshow(field[k], cmap=CMAP, vmin=VMIN, vmax=VMAX)
            ax.set_title(f'{name}  T+{(k + 1) * FRAME_MIN} min', pad=8)
            ax.axis('off')
    fig.colorbar(im, ax=list(axes.flat), fraction=0.02, pad=0.01,
                 label='Brightness temperature (K)')
    fig.suptitle(f'GOES-16 Ch13  test #{INDEX}:  Autoregressive Forecast vs Truth')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / f'index_{INDEX:04d}.png'
    fig.savefig(dst, dpi=DPI)
    plt.close(fig)
    print(f'[done] 已保存 {dst}')


if __name__ == '__main__':
    main()
