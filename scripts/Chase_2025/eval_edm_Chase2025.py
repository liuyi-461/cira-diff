"""Evaluate the official plain-EDM diffusion baseline on the GOES-16 Ch13 test set.

Runs an 18-step autoregressive ensemble forecast with the official pretrained EDM
weights and scores it with the *same* Kelvin-domain protocol as FlowCast v3-A
(threshold 235 K, FSS window 9, 8 ensemble members) so the resulting metrics.json
is directly comparable to zjt's runs/evaluation/metrics.json.

The Metrics class and normalization constants below are copied verbatim from
    zjt/cira-diff/flowcast_v2/datasets/goes/metrics.py
    zjt/cira-diff/flowcast_v2/datasets/goes/normalization.py
The EDM classes (EDMPrecond, edm_sampler, StackedRandomGenerator) and the
autoregressive rollout are copied verbatim from the official
    scripts/Chase_2025/Run_Forecasts_Chase2025.ipynb
so the sampling is faithful to the paper.

Usage (from any cwd):
    python scripts/Chase_2025/eval_edm_Chase2025.py \
        --checkpoint /data1/satcast/edm_plain_diffusion/edm_plain_diffusion/checkpoint.pth \
        --data /data1/satcast/edm_GOES_ch13_test_dataset.zarr \
        --output_dir runs/evaluation
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
# Normalization (verbatim from zjt/.../goes/normalization.py)
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
# Metrics (verbatim from zjt/.../goes/metrics.py)
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
# EDM classes (verbatim from Run_Forecasts_Chase2025.ipynb)
# ---------------------------------------------------------------------------
class EDMPrecond(torch.nn.Module):
    """Karras et al. (2022) preconditioning wrapper adapted for conditional EDM."""

    def __init__(self, generation_channels, model, use_fp16=False, sigma_min=0,
                 sigma_max=float("inf"), sigma_data=0.5):
        super().__init__()
        self.generation_channels = generation_channels
        self.model = model
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    def forward(self, x, sigma, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_noisy = torch.clone(x[:, 0:self.generation_channels])
        x_condition = torch.clone(x[:, self.generation_channels:])
        model_input_images = torch.cat([x_noisy * c_in, x_condition], dim=1)

        F_x = self.model((model_input_images).to(dtype), c_noise.flatten(), return_dict=False)[0]
        # torch.autocast may return fp16 here even though the precond math above is fp32.
        F_x = F_x.to(torch.float32)
        D_x = c_skip * x_noisy + c_out * F_x
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class StackedRandomGenerator:
    """Per-sample seeded generators, adapted from NVlabs/edm."""

    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError(f"Expected first dimension of size {len(self.generators)}, got {size[0]}")
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])


def edm_sampler(net, latents, condition_images, randn_like=torch.randn_like, num_steps=18,
                sigma_min=0.002, sigma_max=80, rho=7, S_churn=0, S_min=0, S_max=float("inf"), S_noise=1):
    """Karras et al. (2022) sampler (Euler + 2nd-order correction), adapted for conditional EDM."""
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        model_input_images = torch.cat([x_hat, condition_images], dim=1)
        with torch.no_grad():
            denoised = net(model_input_images, t_hat).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            model_input_images = torch.cat([x_next, condition_images], dim=1)
            with torch.no_grad():
                denoised = net(model_input_images, t_next).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def rollout(model, condition, ens_size, leads, num_steps, S_churn, sigma_max, rho, device):
    """18-step autoregressive ensemble forecast (sliding 2-frame window).

    Returns forecast frames only, shape [B, ens_size, leads, H, W] in normalized space.
    """
    B, _, H, W = condition.shape
    full = torch.zeros([B, ens_size, leads + 2, H, W])  # first 2 slots hold the condition
    full[:, :, 0:2] = condition.unsqueeze(1).repeat(1, ens_size, 1, 1, 1).cpu()

    for t in tqdm(range(leads), desc="lead", leave=False):
        # A fresh set of per-sample seeds each lead time (same_seed=False in the notebook).
        rnd = StackedRandomGenerator(device, np.arange(0, B * ens_size) + t)
        latents = rnd.randn([B * ens_size, 1, H, W], device=device)

        current = full[:, :, t:t + 2].to(torch.float32).to(device)
        current = current.reshape(B * ens_size, 2, H, W)

        images = edm_sampler(model, latents, current, num_steps=num_steps,
                             S_churn=S_churn, sigma_max=sigma_max, rho=rho)
        full[:, :, t + 2:t + 3] = images.reshape(B, ens_size, 1, H, W).cpu()

    return full[:, :, 2:]


def build_model(checkpoint, device):
    model = UNet2DModel(
        sample_size=256,
        in_channels=3,          # 1 noisy generation channel + 2 condition channels
        out_channels=1,         # a single 10-min forecast
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D",
                          "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return EDMPrecond(1, model)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="raw test zarr with input_images/output_images")
    p.add_argument("--output_dir", default="runs/evaluation")
    p.add_argument("--max_samples", type=int, default=None, help="cap number of test sequences")
    p.add_argument("--index", type=int, default=0, help="starting sample index (for sharding across GPUs)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--samples", type=int, default=8, help="ensemble members")
    p.add_argument("--steps", type=int, default=36, help="EDM sampler num_steps")
    p.add_argument("--S_churn", type=float, default=7.2)
    p.add_argument("--sigma_max", type=float, default=140.0)
    p.add_argument("--rho", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=235.0)
    p.add_argument("--fss_window", type=int, default=9)
    p.add_argument("--use_fp16", action="store_true",
                   help="run the UNet body at fp16 (training used mixed-precision fp16); ~2-3x faster")
    p.add_argument("--save_forecasts", action="store_true",
                   help="also write forecasts_{index}_{count}.npy (float32 Kelvin, "
                        "[sample, lead, member, H, W]) for this shard")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    raw = zarr.open(args.data, mode="r")
    n_total = raw["input_images"].shape[0]
    if args.index < 0 or args.index >= n_total:
        p.error(f"--index {args.index} is outside the dataset (0..{n_total - 1})")
    count = (n_total - args.index) if args.max_samples is None else min(args.max_samples, n_total - args.index)
    leads = raw["output_images"].shape[1]
    assert leads == 18, f"expected 18 lead frames, got {leads}"

    model = build_model(args.checkpoint, device)
    data_range = NORM["kelvin_max"] - NORM["kelvin_min"]
    metrics = {"persistence": Metrics(data_range, args.threshold, args.fss_window),
               "edm": Metrics(data_range, args.threshold, args.fss_window)}
    # Per-lead-time (1..18) metric accumulators, in addition to the pooled ones above.
    metrics_by_lead = {name: [Metrics(data_range, args.threshold, args.fss_window) for _ in range(leads)]
                       for name in metrics}

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Forecasts are stored in test order as [sample, lead, member, H, W] in Kelvin.
    forecasts = (np.zeros((count, leads, args.samples, *raw["output_images"].shape[-2:]),
                          dtype=np.float32) if args.save_forecasts else None)

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
            pred = rollout(model, condition, args.samples, leads, args.steps,
                           args.S_churn, args.sigma_max, args.rho, device)
            pred_k = to_kelvin(pred.numpy())          # [B, S, T, H, W] in Kelvin
            metrics["edm"].update(pred_k.transpose(1, 0, 2, 3, 4), truth)  # (S,B,T,H,W)

            local = start - args.index
            batch = stop - start
            if forecasts is not None:
                forecasts[local:local + batch] = pred_k.transpose(0, 2, 1, 3, 4)  # (B,T,S,H,W)
            for t in range(leads):
                metrics_by_lead["persistence"][t].update(persistence[:, t:t + 1][None], truth[:, t:t + 1])
                metrics_by_lead["edm"][t].update(pred_k[:, :, t:t + 1].transpose(1, 0, 2, 3, 4),
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
            samples=args.samples,
            steps=args.steps,
            S_churn=args.S_churn,
            sigma_max=args.sigma_max,
            rho=args.rho,
            seed=args.seed,
            threshold=args.threshold,
            fss_window=args.fss_window,
            use_fp16=args.use_fp16,
            save_forecasts=args.save_forecasts,
            index=args.index,
            split="test",
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
