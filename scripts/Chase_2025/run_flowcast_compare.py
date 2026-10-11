"""Sharded full-eval of zjt's trained FlowCast (VAE-latent conditional-flow model)
on the GOES-16 Ch13 test set, scored with the same Kelvin-domain protocol as
eval_edm_Chase2025.py (threshold 235 K, FSS window 9).

This script does NOT modify any of zjt's code. It imports zjt's model loaders from
zjt/cira-diff/flowcast_v2/datasets/goes/{runtime,normalization}.py, and the
`decode`/`forecast`/`latent_shape` helpers below are copied verbatim from zjt's
    zjt/cira-diff/flowcast_v2/scripts/evaluate.py
so the sampling is byte-faithful to zjt's evaluation protocol.

Sharding matches eval_edm_Chase2025.py: each process handles samples
[index, index + count) where count = min(max_samples, n_total - index). It writes
    forecasts_{index}_{count}.npy   [sample, lead, member=8, H, W]  float32 Kelvin
and a metrics.json carrying persistence + flowcast metrics (aggregate + per-lead)
plus `_accumulator` / `_accumulator_by_lead` raw sums for merge_metrics.py.

Usage (4 shards of 256):
    python scripts/Chase_2025/run_flowcast_compare.py \
        --output_dir runs/evaluation/flowcast_full/shard_0 --index 0 --max_samples 256 --device cuda:0
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import zarr
from scipy.ndimage import uniform_filter, gaussian_filter
from tqdm import tqdm

ZJT = Path("/home/group1/26fall_aiclass/zjt/cira-diff/flowcast_v2")
sys.path.insert(0, str(ZJT))

from datasets.goes.runtime import load_vae, load_flowcast, read_config, sha256  # noqa: E402
from datasets.goes.normalization import to_kelvin, normalize_to_neg_one_one, denormalize_to_kelvin  # noqa: E402


# ---------------------------------------------------------------------------
# Metrics (verbatim from eval_edm_Chase2025.py; identical to zjt's Metrics plus
# the `state()` raw-sum accessor that lets sharded runs be merged exactly).
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
# Verbatim from zjt/.../scripts/evaluate.py
# ---------------------------------------------------------------------------
def decode(vae, latent, norm, frame_batch=8):
    b, t, h, w, c = latent.shape
    frames = latent.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
    decoded = torch.cat([vae.decode(chunk).sample for chunk in frames.split(frame_batch)])
    return denormalize_to_kelvin(decoded[:, 0], norm).reshape(b, t, *decoded.shape[-2:])


def forecast(model, condition, target_shape, samples, steps, horizon):
    # Roll the model out to the full target horizon; horizon==1 for multi-step models.
    condition = model.normalize(condition)
    for _ in range(samples):
        cond, states = condition, []
        for _ in range(horizon):
            state = torch.randn(target_shape, device=cond.device)
            for step in range(steps):
                time = torch.full((len(state),), step / steps, device=state.device)
                state = state + model(time, state, cond) / steps
            states.append(model.denormalize(state))
            # Feed the prediction back, keeping the lag window fixed at lag_time frames.
            cond = torch.cat([cond, state], dim=1)[:, -condition.shape[1]:]
        yield torch.cat(states, dim=1)


def latent_shape(latent, key):
    _, t, c, h, w = latent[key].shape
    return (t, h, w, c)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vae_checkpoint", default=str(ZJT / "runs/vae/vae_1005_150523/models/early_stopping_model.pt"))
    p.add_argument("--vae_config", default=str(ZJT / "runs/vae/vae_1005_150523/config_snapshot.yaml"))
    p.add_argument("--cfm_checkpoint", default=str(ZJT / "runs/cfm/cfm_1007_010810/models/early_stopping_model.pt"))
    p.add_argument("--cfm_config", default=str(ZJT / "runs/cfm/cfm_1007_010810/config_snapshot.yaml"))
    p.add_argument("--latent_path", default=str(ZJT / "runs/latent/test.zarr"))
    p.add_argument("--data", default=str(ZJT.parent / "datasets/satcast/edm_GOES_ch13_test_dataset.zarr"))
    p.add_argument("--output_dir", default="runs/evaluation/flowcast_full/shard_0")
    p.add_argument("--max_samples", type=int, default=None, help="cap number of test sequences")
    p.add_argument("--index", type=int, default=0, help="starting sample index (for sharding across GPUs)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--samples", type=int, default=8, help="ensemble members")
    p.add_argument("--steps", type=int, default=10, help="CFM Euler steps")
    p.add_argument("--threshold", type=float, default=235.0)
    p.add_argument("--fss_window", type=int, default=9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    vae, norm = load_vae(args.vae_checkpoint, args.vae_config, device)
    raw = zarr.open(args.data, mode="r")
    latent = zarr.open(args.latent_path, mode="r")
    # Provenance guard (same checks zjt's evaluate.py enforces).
    assert latent.attrs.get("complete") is True and latent.attrs.get("encoding") == "mode"
    assert latent.attrs.get("vae_sha256") == sha256(args.vae_checkpoint)
    assert latent.attrs.get("normalization") == norm
    assert Path(latent.attrs["source_path"]).resolve() == Path(args.data).resolve()

    model = load_flowcast(args.cfm_checkpoint, args.cfm_config,
                          latent_shape(latent, "input_images"),
                          latent_shape(latent, "output_images"), device)

    n_total = raw["input_images"].shape[0]
    if args.index < 0 or args.index >= n_total:
        p.error(f"--index {args.index} is outside the dataset (0..{n_total - 1})")
    count = (n_total - args.index) if args.max_samples is None else min(args.max_samples, n_total - args.index)
    leads = raw["output_images"].shape[1]
    assert leads == 18, f"expected 18 lead frames, got {leads}"
    h, w = raw["output_images"].shape[-2:]

    data_range = norm["kelvin_max"] - norm["kelvin_min"]
    metrics = {"persistence": Metrics(data_range, args.threshold, args.fss_window),
               "flowcast": Metrics(data_range, args.threshold, args.fss_window)}
    metrics_by_lead = {name: [Metrics(data_range, args.threshold, args.fss_window) for _ in range(leads)]
                       for name in metrics}

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    forecasts = np.zeros((count, leads, args.samples, h, w), dtype=np.float32)

    with torch.inference_mode():
        for start in tqdm(range(args.index, args.index + count, args.batch_size), desc="batch"):
            stop = min(start + args.batch_size, args.index + count)
            x = np.asarray(raw["input_images"][start:stop], dtype=np.float32)
            y = np.asarray(raw["output_images"][start:stop], dtype=np.float32)
            b = y.shape[0]

            truth = to_kelvin(y, norm)
            persistence = np.repeat(to_kelvin(x[:, -1:], norm), leads, axis=1)
            metrics["persistence"].update(persistence[None], truth)

            # FlowCast autoregressive rollout in latent space.
            condition = torch.from_numpy(np.asarray(latent["input_images"][start:stop])).permute(0, 1, 3, 4, 2).to(device)
            target_shape = (b, *latent_shape(latent, "output_images"))
            horizon = -(-leads // latent_shape(latent, "output_images")[0])
            preds = np.stack([decode(vae, z, norm).cpu().numpy()
                              for z in forecast(model, condition, target_shape, args.samples, args.steps, horizon)])
            preds = preds[:, :, :leads]                      # [member, B, lead, H, W]
            preds = preds.transpose(1, 2, 0, 3, 4)           # [B, lead, member, H, W]
            metrics["flowcast"].update(preds.transpose(2, 0, 1, 3, 4), truth)  # (member,B,lead,H,W)

            local = start - args.index
            forecasts[local:local + b] = preds
            for t in range(leads):
                metrics_by_lead["persistence"][t].update(persistence[:, t:t + 1][None], truth[:, t:t + 1])
                metrics_by_lead["flowcast"][t].update(preds[:, t:t + 1].transpose(2, 0, 1, 3, 4),
                                                      truth[:, t:t + 1])

    npy_path = out / f"forecasts_{args.index}_{count}.npy"
    np.save(npy_path, forecasts)
    print(f"wrote {npy_path} ({forecasts.shape}, {forecasts.dtype})")

    report = dict(
        arguments=dict(
            vae_checkpoint=str(Path(args.vae_checkpoint).resolve()),
            cfm_checkpoint=str(Path(args.cfm_checkpoint).resolve()),
            vae_config=str(Path(args.vae_config).resolve()),
            cfm_config=str(Path(args.cfm_config).resolve()),
            latent_path=str(Path(args.latent_path).resolve()),
            data=str(Path(args.data).resolve()),
            output_dir=str(out.resolve()),
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            samples=args.samples,
            steps=args.steps,
            threshold=args.threshold,
            fss_window=args.fss_window,
            seed=args.seed,
            index=args.index,
            split="test",
            model="flowcast_vae_cfm",
        ),
        normalization=norm,
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
