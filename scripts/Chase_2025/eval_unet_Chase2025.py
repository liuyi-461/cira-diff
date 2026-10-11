"""Evaluate the plain deterministic UNet baseline on the GOES-16 Ch13 test set.

Runs an 18-step autoregressive forecast with the pretrained vanilla UNet
(diffusers UNet2DModel, in_channels=2 -> out_channels=1) and scores it with the
*same* Kelvin-domain protocol as eval_edm_Chase2025.py (threshold 235 K, FSS
window 9) so the resulting metrics.json is directly comparable. Unlike EDM the
UNet is deterministic: a single forward pass per lead time, no diffusion steps
and no ensemble members.

The Metrics class, normalization constants, data loading and report format are
copied verbatim from scripts/Chase_2025/eval_edm_Chase2025.py. The UNet
rollout is the notebook's `run_forecast_unet` (input 2 frames, timestep=0,
predict 1 frame, slide the window).

Usage (from any cwd):
    python scripts/Chase_2025/eval_unet_Chase2025.py \
        --checkpoint /data1/satcast/unet_vanilla/unet_vanilla \
        --data /data1/satcast/edm_GOES_ch13_test_dataset.zarr \
        --output_dir runs/evaluation/unet_shard_0 --index 0 --max_samples 256
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import zarr
from diffusers import UNet2DModel
from scipy.ndimage import uniform_filter, gaussian_filter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Normalization (verbatim from eval_edm_Chase2025.py)
# ---------------------------------------------------------------------------
NORM = dict(
    mean=279.0699,
    std=19.3297,
    kelvin_min=159.99592773437502,
    kelvin_max=340.003758984375,
)


def to_kelvin(x, config=NORM):
    return x * config["std"] + config["mean"]


# ---------------------------------------------------------------------------
# Metrics (verbatim from eval_edm_Chase2025.py)
# ---------------------------------------------------------------------------
class Metrics:
    def __init__(self, data_range, threshold=235.0, window=9):
        if data_range <= 0 or window <= 0 or window % 2 == 0:
            raise ValueError("Positive data range and odd positive FSS window required")
        self.data_range, self.threshold, self.window = data_range, threshold, window
        self.n = self.frames = self.se = self.ae = self.crps = self.ssim = 0.0
        self.hits = self.union = self.fss_error = self.fss_ref = 0.0

    def update(self, ensemble, target):
        # (S,B,T,H,W) and (B,T,H,W), all in Kelvin.
        ensemble, target = np.asarray(ensemble, dtype=np.float64), np.asarray(target, dtype=np.float64)
        if ensemble.shape[1:] != target.shape or not np.isfinite(ensemble).all() or not np.isfinite(target).all():
            raise ValueError("Metric inputs have incompatible shapes or nonfinite values")
        pred = ensemble.mean(axis=0)
        error = pred - target
        self.se += np.square(error).sum(); self.ae += np.abs(error).sum(); self.n += target.size
        ordered = np.sort(ensemble, axis=0)
        s = len(ensemble)
        weights = (2 * np.arange(s) - s + 1).reshape((s,) + (1,) * target.ndim)
        self.crps += (np.abs(ensemble - target).mean(axis=0) - (ordered * weights).sum(axis=0) / (s * s)).sum()
        event_p, event_t = pred <= self.threshold, target <= self.threshold
        self.hits += np.logical_and(event_p, event_t).sum()
        self.union += np.logical_or(event_p, event_t).sum()
        for p, t, ep, et in zip(pred.reshape(-1, *pred.shape[-2:]), target.reshape(-1, *target.shape[-2:]),
                                event_p.reshape(-1, *pred.shape[-2:]), event_t.reshape(-1, *target.shape[-2:])):
            u, v = gaussian_filter(p, 1.5), gaussian_filter(t, 1.5)
            vp = np.maximum(gaussian_filter(p * p, 1.5) - u * u, 0)
            vt = np.maximum(gaussian_filter(t * t, 1.5) - v * v, 0)
            cov = gaussian_filter(p * t, 1.5) - u * v
            c1, c2 = (0.01 * self.data_range) ** 2, (0.03 * self.data_range) ** 2
            self.ssim += np.mean(((2 * u * v + c1) * (2 * cov + c2)) / ((u * u + v * v + c1) * (vp + vt + c2)))
            self.frames += 1
            fp, ft = uniform_filter(ep.astype(float), self.window), uniform_filter(et.astype(float), self.window)
            self.fss_error += np.square(fp - ft).sum()
            self.fss_ref += (fp * fp + ft * ft).sum()

    def result(self):
        if not self.n:
            raise ValueError("No samples evaluated")
        mse = self.se / self.n
        return dict(mse_k2=mse, mae_k=self.ae / self.n, rmse_k=float(np.sqrt(mse)),
                    psnr_db=float(10 * np.log10(self.data_range ** 2 / mse)) if mse else None,
                    ssim=self.ssim / self.frames, crps_k=self.crps / self.n,
                    csi=self.hits / self.union if self.union else None,
                    fss=1 - self.fss_error / self.fss_ref if self.fss_ref else None,
                    pixels=int(self.n))

    def state(self):
        """Raw running sums, so sharded runs can be merged exactly."""
        return dict(n=self.n, frames=self.frames, se=self.se, ae=self.ae, crps=self.crps,
                    ssim=self.ssim, hits=self.hits, union=self.union,
                    fss_error=self.fss_error, fss_ref=self.fss_ref)


# ---------------------------------------------------------------------------
# Deterministic UNet rollout (run_forecast_unet from Run_Forecasts_Chase2025.ipynb)
# ---------------------------------------------------------------------------
def rollout_unet(model, condition, leads, device):
    """18-step autoregressive forecast (sliding 2-frame window), one forward per step.

    Returns forecast frames only, shape [B, 1, leads, H, W] in normalized space.
    """
    B, _, H, W = condition.shape
    full = torch.zeros([B, leads + 2, H, W])
    full[:, 0:2] = condition.cpu()
    timesteps = torch.zeros(B, device=device)  # deterministic model: fixed zero timestep
    for t in tqdm(range(leads), desc="lead", leave=False):
        current = full[:, t:t + 2].to(torch.float32).to(device)
        images = model(current, timesteps, return_dict=False)[0]  # [B, 1, H, W]
        full[:, t + 2] = images[:, 0].to(torch.float32).cpu()
    return full[:, 2:].unsqueeze(1)  # [B, 1, leads, H, W]


def build_unet(checkpoint, device):
    model = UNet2DModel.from_pretrained(checkpoint)
    model.to(device).eval()
    return model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="diffusers UNet2DModel directory (config.json + diffusion_pytorch_model.safetensors)")
    p.add_argument("--data", required=True, help="raw test zarr with input_images/output_images")
    p.add_argument("--output_dir", default="runs/evaluation")
    p.add_argument("--max_samples", type=int, default=None, help="cap number of test sequences")
    p.add_argument("--index", type=int, default=0, help="starting sample index (for sharding across GPUs)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--threshold", type=float, default=235.0)
    p.add_argument("--fss_window", type=int, default=9)
    p.add_argument("--use_fp16", action="store_true", help="run the UNet body at fp16")
    p.add_argument("--save_forecasts", action="store_true",
                   help="also write forecasts_{index}_{count}.npy (float32 Kelvin, "
                        "[sample, lead, member=1, H, W]) for this shard")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    raw = zarr.open(args.data, mode="r")
    n_total = raw["input_images"].shape[0]
    if args.index < 0 or args.index >= n_total:
        p.error(f"--index {args.index} is outside the dataset (0..{n_total - 1})")
    count = (n_total - args.index) if args.max_samples is None else min(args.max_samples, n_total - args.index)
    leads = raw["output_images"].shape[1]
    assert leads == 18, f"expected 18 lead frames, got {leads}"

    model = build_unet(args.checkpoint, device)
    data_range = NORM["kelvin_max"] - NORM["kelvin_min"]
    metrics = {"persistence": Metrics(data_range, args.threshold, args.fss_window),
               "unet": Metrics(data_range, args.threshold, args.fss_window)}
    metrics_by_lead = {name: [Metrics(data_range, args.threshold, args.fss_window) for _ in range(leads)]
                       for name in metrics}

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    forecasts = (np.zeros((count, leads, 1, *raw["output_images"].shape[-2:]), dtype=np.float32)
                 if args.save_forecasts else None)

    autocast = torch.autocast(device_type="cuda", dtype=torch.float16,
                              enabled=args.use_fp16 and device.type == "cuda")
    with torch.inference_mode(), autocast:
        for start in tqdm(range(args.index, args.index + count, args.batch_size), desc="batch"):
            stop = min(start + args.batch_size, args.index + count)
            x = np.asarray(raw["input_images"][start:stop], dtype=np.float32)
            y = np.asarray(raw["output_images"][start:stop], dtype=np.float32)

            truth = to_kelvin(y)
            persistence = np.repeat(to_kelvin(x[:, -1:]), y.shape[1], axis=1)
            metrics["persistence"].update(persistence[None], truth)

            condition = torch.from_numpy(x).to(device)
            pred = rollout_unet(model, condition, leads, device)  # [B, 1, leads, H, W] normalized
            pred_k = to_kelvin(pred.numpy())                      # [B, 1, leads, H, W] Kelvin
            metrics["unet"].update(pred_k.transpose(1, 0, 2, 3, 4), truth)  # (1,B,T,H,W)

            local = start - args.index
            batch = stop - start
            if forecasts is not None:
                forecasts[local:local + batch] = pred_k.transpose(0, 2, 1, 3, 4)  # (B,T,1,H,W)
            for t in range(leads):
                metrics_by_lead["persistence"][t].update(persistence[:, t:t + 1][None], truth[:, t:t + 1])
                metrics_by_lead["unet"][t].update(pred_k[:, :, t:t + 1].transpose(1, 0, 2, 3, 4),
                                                  truth[:, t:t + 1])

    if forecasts is not None:
        npy_path = out / f"forecasts_{args.index}_{count}.npy"
        np.save(npy_path, forecasts)
        print(f"wrote {npy_path} ({forecasts.shape}, {forecasts.dtype})")

    report = dict(
        arguments=dict(
            checkpoint=str(Path(args.checkpoint).resolve()),
            data=str(Path(args.data).resolve()),
            output_dir=str(out.resolve()),
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            threshold=args.threshold,
            fss_window=args.fss_window,
            use_fp16=args.use_fp16,
            save_forecasts=args.save_forecasts,
            index=args.index,
            split="test",
            model="unet2d_deterministic",
        ),
        normalization=NORM,
        samples_evaluated=count,
        metric_definition="Kelvin; CSI/FSS on ensemble-mean Tb <= threshold; reflected spatial windows; "
                          "PSNR null for zero MSE; CSI/FSS null for no events",
        metrics={k: v.result() for k, v in metrics.items()},
        metrics_by_lead={k: [v.result() for v in vs] for k, vs in metrics_by_lead.items()},
        _accumulator={k: v.state() for k, v in metrics.items()},  # stripped by merge_metrics.py
        _accumulator_by_lead={k: [v.state() for v in vs] for k, vs in metrics_by_lead.items()},
    )
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
