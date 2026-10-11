"""FlowCast (VAE-latent) vs EDM (pixel-space) blur comparison, from dcm's own
saved forecasts. Produces per-sample side-by-side figures for the last four lead
frames (150/160/170/180 min) plus a quantitative sharpness summary.

Inputs (all Kelvin float32, test order):
    runs/evaluation/flowcast_compare/flowcast_forecasts.npy  [sample, lead, member, H, W]
    runs/evaluation/flowcast_compare/{vae_recon,persistence,truth,input}.npy
    runs/evaluation/edm_10/forecasts_0_10.npy                [sample, lead, member, H, W]

Sharpness proxy: variance of the Laplacian (higher = sharper) and mean gradient
magnitude, averaged over 10 samples x leads 150-180 min. Single-member vs
ensemble-mean sharpness isolates ensemble-averaging blur from VAE/model blur.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import laplace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEADS = [14, 15, 16, 17]                       # 0-indexed -> 150/160/170/180 min
LEAD_LABELS = ["150 min", "160 min", "170 min", "180 min"]
VMIN, VMAX = 200.0, 300.0


def sharpness(img):
    """(variance of Laplacian, mean gradient magnitude); higher == sharper."""
    f = np.asarray(img, dtype=np.float64)
    lap = laplace(f)
    gy, gx = np.gradient(f)
    return float(lap.var()), float(np.sqrt(gx ** 2 + gy ** 2).mean())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fc_dir", default="runs/evaluation/flowcast_compare")
    p.add_argument("--edm_npy", default="runs/evaluation/edm_10/forecasts_0_10.npy")
    p.add_argument("--out_dir", default="runs/evaluation/flowcast_compare/figs")
    args = p.parse_args(argv)

    fc_dir = Path(args.fc_dir)
    flowcast = np.load(fc_dir / "flowcast_forecasts.npy")       # (N,18,8,H,W)
    vae = np.load(fc_dir / "vae_recon.npy")                     # (N,18,H,W)
    persistence = np.load(fc_dir / "persistence.npy")
    truth = np.load(fc_dir / "truth.npy")
    inp = np.load(fc_dir / "input.npy")                         # (N,2,H,W)
    edm = np.load(args.edm_npy)                                 # (N,18,8,H,W)

    n = truth.shape[0]
    assert flowcast.shape[0] == edm.shape[0] == n, "FlowCast/EDM sample counts differ"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- per-sample figures -------------------------------------------------
    for s in range(n):
        fig, axes = plt.subplots(len(LEADS), 5, figsize=(5 * 2.1, len(LEADS) * 2.2),
                                 constrained_layout=True)
        for r, lead in enumerate(LEADS):
            last_obs = inp[s, -1] if r == 0 else None
            panels = [
                ("last obs", inp[s, -1]),
                ("truth", truth[s, lead]),
                ("VAE recon", vae[s, lead]),
                ("FlowCast mean", flowcast[s, lead].mean(0)),
                ("EDM mean", edm[s, lead].mean(0)),
            ]
            for c, (name, img) in enumerate(panels):
                ax = axes[r, c]
                lap, _ = sharpness(img)
                ax.imshow(img, cmap="gray", vmin=VMIN, vmax=VMAX)
                ax.set_title(f"{name}\nLapVar={lap:.2e}", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            axes[r, 0].set_ylabel(LEAD_LABELS[r], fontsize=9)
        fig.suptitle(f"sample {s}  (gray 200-300 K; LapVar higher = sharper)", fontsize=10)
        fig.savefig(out / f"sample_{s:03d}.png", dpi=110)
        plt.close(fig)

    # --- quantitative sharpness summary ------------------------------------
    keys = {
        "truth": truth,
        "persistence": persistence,
        "VAE_recon": vae,
        "FlowCast_mean": flowcast.mean(2),
        "FlowCast_member0": flowcast[:, :, 0],
        "EDM_mean": edm.mean(2),
        "EDM_member0": edm[:, :, 0],
    }
    summary = {}
    for name, arr in keys.items():
        lap, grad = 0.0, 0.0
        for lead in LEADS:
            for s in range(n):
                l, g = sharpness(arr[s, lead])
                lap += l; grad += g
        m = n * len(LEADS)
        summary[name] = dict(lap_var=lap / m, grad_mag=grad / m)

    (out / "sharpness.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    # Bar chart of variance-of-Laplacian (sharpness).
    names = list(summary)
    laps = [summary[k]["lap_var"] for k in names]
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#333"] + ["#888"] + ["#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e", "#ff7f0e"]
    bars = ax.bar(names, laps, color=["#555", "#999", "#1f77b4", "#5aa7de", "#ff7f0e", "#ffa64d", "#ffc48c"])
    ax.set_ylabel("variance of Laplacian (sharpness)")
    ax.set_title("Sharpness of 10-sample forecasts, leads 150-180 min (higher = sharper)")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, laps):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2e}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sharpness.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out}/sample_*.png ({n} figures) + sharpness.json + sharpness.png")


if __name__ == "__main__":
    main()
