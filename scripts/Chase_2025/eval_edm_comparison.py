"""
Unified EDM Evaluation: Official vs EXP-011 Reproduced EDM
==========================================================
Single evaluation pipeline, swaps only the checkpoint path.
Uses canonical sampling protocol from Run_Forecasts_Chase2025.ipynb CELL 9.
"""

import os, sys, json, gc, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel
from tqdm import tqdm


# ============================================================
# EDM Preconditioning + Sampler (from train_edm_Chase2025.py)
# ============================================================

class EDMPrecond(torch.nn.Module):
    def __init__(self, generation_channels, model, use_fp16=False,
                 sigma_min=0, sigma_max=float('inf'), sigma_data=0.5):
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
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        x_noisy = torch.clone(x[:, 0:self.generation_channels])
        x_condition = torch.clone(x[:, self.generation_channels:])
        model_input_images = torch.cat([x_noisy * c_in, x_condition], dim=1)
        F_x = self.model((model_input_images).to(dtype), c_noise.flatten(), return_dict=False)[0]
        assert F_x.dtype == dtype
        D_x = c_skip * x_noisy + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def edm_sampler(net, latents, condition_images, num_steps=36,
                sigma_min=0.002, sigma_max=140, rho=4,
                S_churn=7.2, S_min=0, S_max=float('inf'), S_noise=1):
    device = latents.device
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
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
    return x_next.to(torch.float32)


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)


# ============================================================
# Metrics (from eval_unet_comparison.py, reused)
# ============================================================

def ssim_pytorch(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    if img1.dim() == 3:
        img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=(1, 2, 3))


def psnr_pytorch(pred, target):
    mse = F.mse_loss(pred, target, reduction='none').mean(dim=(1, 2, 3))
    return 10 * torch.log10(1.0 / (mse + 1e-12))


# ============================================================
# Dataset
# ============================================================

class TestZarrDataset(Dataset):
    def __init__(self, zarr_store):
        self.data = zarr.open(zarr_store, mode='r')
        self.input_images = self.data['input_images']   # (N, 2, 256, 256)
        self.output_images = self.data['output_images']  # (N, 18, 256, 256)
        self.num_samples = self.input_images.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        inputs = torch.from_numpy(self.input_images[idx].astype(np.float32))
        targets = torch.from_numpy(self.output_images[idx].astype(np.float32))
        return inputs, targets


# ============================================================
# Model loading (two formats: diffusers safetensors vs PyTorch .pth)
# ============================================================

def build_unet():
    return UNet2DModel(
        sample_size=256,
        in_channels=3,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D",
                          "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                        "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


def load_edm_model(ckpt_path, ckpt_format, device='cuda'):
    if ckpt_format == 'diffusers':
        base = UNet2DModel.from_pretrained(ckpt_path)
    elif ckpt_format == 'pth':
        base = build_unet()
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        missing, unexpected = base.load_state_dict(ckpt['model_state_dict'], strict=True)
        assert len(missing) == 0 and len(unexpected) == 0, f"Load failed: missing={missing}, unexpected={unexpected}"
        print(f"  Loaded official checkpoint: epoch={ckpt['epoch']}, step={ckpt['step']}")
    else:
        raise ValueError(f"Unknown ckpt_format: {ckpt_format}")

    wrapped = EDMPrecond(generation_channels=1, model=base, use_fp16=False,
                         sigma_min=0, sigma_max=float('inf'), sigma_data=0.5)
    wrapped.to(device)
    wrapped.eval()
    return wrapped


# ============================================================
# Single-step evaluation (teacher-forced, with ensemble)
# ============================================================

def evaluate_single_step_ensemble(model_edm, dataset, device='cuda',
                                   ens_size=10, num_steps=36,
                                   sigma_min=0.002, sigma_max=140, rho=4,
                                   S_churn=7.2, S_min=0, S_max=float('inf'), S_noise=1,
                                   max_samples=None, base_seed=0, batch_size=1):
    """
    Single-step EDM denoising with ensemble.
    For each sample, generates ens_size members with different noise seeds,
    then reports both single-member (mean of all members) and ensemble-mean metrics.

    Noise seeds: sample_idx * ens_size + member_idx + base_seed
    """
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    single_member = {'mse': [], 'mae': [], 'ssim': [], 'psnr': []}
    ensemble_mean = {'mse': [], 'mae': [], 'ssim': [], 'psnr': []}

    model_edm.eval()
    all_preds_per_sample = []  # list of (ens_size, 1, 256, 256) for ensemble variance calc

    for idx in tqdm(range(n), desc=f"Single-step EDM ensemble (ens={ens_size}, n={n})", mininterval=5):
        inputs, targets = dataset[idx]
        cond = inputs.unsqueeze(0).to(device)  # (1, 2, 256, 256)
        target_t1 = targets[0:1].to(device)   # (1, 1, 256, 256)

        seeds = [base_seed + idx * ens_size + m for m in range(ens_size)]
        rnd = StackedRandomGenerator(device, seeds)
        latents = rnd.randn([ens_size, 1, 256, 256], device=device)
        cond_repeated = cond.repeat(ens_size, 1, 1, 1)  # (ens_size, 2, 256, 256)

        with torch.no_grad():
            preds = edm_sampler(model_edm, latents, cond_repeated,
                                num_steps=num_steps, sigma_min=sigma_min,
                                sigma_max=sigma_max, rho=rho,
                                S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise)

        all_preds_per_sample.append(preds.cpu())

        target_expanded = target_t1.repeat(ens_size, 1, 1, 1)

        mse_ind = F.mse_loss(preds.float(), target_expanded.float(), reduction='none').mean(dim=(1, 2, 3)).cpu().numpy()
        mae_ind = F.l1_loss(preds.float(), target_expanded.float(), reduction='none').mean(dim=(1, 2, 3)).cpu().numpy()
        ssim_ind = ssim_pytorch(preds.float(), target_expanded.float()).cpu().numpy()
        psnr_ind = psnr_pytorch(preds.float(), target_expanded.float()).cpu().numpy()

        single_member['mse'].append(mse_ind.mean())
        single_member['mae'].append(mae_ind.mean())
        single_member['ssim'].append(ssim_ind.mean())
        single_member['psnr'].append(psnr_ind.mean())

        ensemble_pred = preds.mean(dim=0, keepdim=True)  # (1, 1, 256, 256)
        mse_ens = F.mse_loss(ensemble_pred.float(), target_t1.float()).item()
        mae_ens = F.l1_loss(ensemble_pred.float(), target_t1.float()).item()
        ssim_ens = ssim_pytorch(ensemble_pred.float(), target_t1.float()).item()
        psnr_ens = psnr_pytorch(ensemble_pred.float(), target_t1.float()).item()

        ensemble_mean['mse'].append(mse_ens)
        ensemble_mean['mae'].append(mae_ens)
        ensemble_mean['ssim'].append(ssim_ens)
        ensemble_mean['psnr'].append(psnr_ens)

    for d in [single_member, ensemble_mean]:
        for k in d:
            d[k] = np.array(d[k])

    return single_member, ensemble_mean, all_preds_per_sample


# ============================================================
# 18-step autoregressive rollout with EDM ensemble
# ============================================================

def rollout_edm_ensemble(model_edm, init_inputs, num_rollout_steps=18, ens_size=10,
                          device='cuda', base_seed=0,
                          num_steps=36, sigma_min=0.002, sigma_max=140, rho=4,
                          S_churn=7.2, S_min=0, S_max=float('inf'), S_noise=1):
    """
    Autoregressive rollout with EDM ensemble.
    Each forecast step generates ens_size members, uses ENSEMBLE MEAN as next condition.
    Same seed protocol as Run_Forecasts_Chase2025: seeds += time_offset per step.

    Returns:
        all_preds: (num_rollout_steps, ens_size, 1, H, W) ensemble members at each step
        ensemble_means: (num_rollout_steps, 1, H, W) ensemble mean at each step
    """
    current = init_inputs.unsqueeze(0).to(device)  # (1, 2, H, W)
    all_preds = []
    ensemble_means = []

    for step in range(num_rollout_steps):
        time_offset = step * 10000  # large offset so seeds don't collide across steps
        seeds = [base_seed + time_offset + m for m in range(ens_size)]
        rnd = StackedRandomGenerator(device, seeds)
        latents = rnd.randn([ens_size, 1, 256, 256], device=device)
        cond_repeated = current.repeat(ens_size, 1, 1, 1)

        with torch.no_grad():
            preds = edm_sampler(model_edm, latents, cond_repeated,
                                num_steps=num_steps, sigma_min=sigma_min,
                                sigma_max=sigma_max, rho=rho,
                                S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise)

        ens_mean = preds.mean(dim=0, keepdim=True)  # (1, 1, H, W)
        all_preds.append(preds.cpu())
        ensemble_means.append(ens_mean.cpu())

        current = torch.cat([current[:, 1:], ens_mean], dim=1)  # (1, 2, H, W)

    return torch.stack(all_preds), torch.stack(ensemble_means)


def evaluate_rollout_ensemble(model_edm, dataset, num_rollout_steps=18,
                              ens_size=10, device='cuda', max_samples=128,
                              base_seed=0, **sampling_kwargs):
    """
    Full autoregressive rollout evaluation with EDM ensemble.
    Per-lead-time metrics for both single-member and ensemble-mean.
    """
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    per_lt_single = {lt: {'mse': [], 'mae': [], 'ssim': [], 'psnr': []} for lt in range(1, num_rollout_steps + 1)}
    per_lt_ens = {lt: {'mse': [], 'mae': [], 'ssim': [], 'psnr': []} for lt in range(1, num_rollout_steps + 1)}

    model_edm.eval()
    t0 = time.time()

    for idx in tqdm(range(n), desc=f"Rollout EDM ensemble (n={n}, steps={num_rollout_steps}, ens={ens_size})", mininterval=5):
        inputs, targets = dataset[idx]

        all_preds, ensemble_means = rollout_edm_ensemble(
            model_edm, inputs, num_rollout_steps=num_rollout_steps,
            ens_size=ens_size, device=device, base_seed=base_seed + idx * 1000,
            **sampling_kwargs
        )

        for lt in range(num_rollout_steps):
            target = targets[lt:lt+1]  # (1, H, W)
            preds_lt = all_preds[lt]   # (ens_size, 1, H, W)
            ens_lt = ensemble_means[lt]  # (1, 1, H, W)

            target_rep = target.unsqueeze(0).repeat(ens_size, 1, 1, 1).float()
            preds_lt_f = preds_lt.float()

            mse_s = F.mse_loss(preds_lt_f, target_rep, reduction='none').mean(dim=(1, 2, 3)).mean().item()
            mae_s = F.l1_loss(preds_lt_f, target_rep, reduction='none').mean(dim=(1, 2, 3)).mean().item()
            ssim_s = ssim_pytorch(preds_lt_f, target_rep).mean().item()
            psnr_s = psnr_pytorch(preds_lt_f, target_rep).mean().item()

            per_lt_single[lt + 1]['mse'].append(mse_s)
            per_lt_single[lt + 1]['mae'].append(mae_s)
            per_lt_single[lt + 1]['ssim'].append(ssim_s)
            per_lt_single[lt + 1]['psnr'].append(psnr_s)

            ens_f = ens_lt.float()
            target_f = target.unsqueeze(0).float()
            per_lt_ens[lt + 1]['mse'].append(F.mse_loss(ens_f, target_f).item())
            per_lt_ens[lt + 1]['mae'].append(F.l1_loss(ens_f, target_f).item())
            per_lt_ens[lt + 1]['ssim'].append(ssim_pytorch(ens_f, target_f).item())
            per_lt_ens[lt + 1]['psnr'].append(psnr_pytorch(ens_f, target_f).item())

    elapsed = time.time() - t0
    print(f"\nRollout done in {elapsed/60:.1f} min ({elapsed/n:.1f}s per sample)")

    for d in [per_lt_single, per_lt_ens]:
        for lt in d:
            for k in d[lt]:
                d[lt][k] = np.array(d[lt][k])

    return per_lt_single, per_lt_ens


# ============================================================
# Printing
# ============================================================

def print_single_step_comparison(name_a, sm_a, em_a, name_b, sm_b, em_b):
    print("\n" + "=" * 90)
    print(f"  SINGLE-STEP EDM COMPARISON (canonical sampler, same noise seeds)")
    print("=" * 90)
    print(f"  {'Metric':>8s} | {name_a+' (single)':>20s} {name_b+' (single)':>20s} {'Δ':>10s} | {name_a+' (ens)':>20s} {name_b+' (ens)':>20s} {'Δ':>10s}")
    print("-" * 90)
    for k in ['mse', 'mae', 'ssim', 'psnr']:
        sa, sb = sm_a[k].mean(), sm_b[k].mean()
        ea, eb = em_a[k].mean(), em_b[k].mean()
        print(f"  {k.upper():>8s} | {sa:>20.6f} {sb:>20.6f} {sb-sa:>+10.6f} | {ea:>20.6f} {eb:>20.6f} {eb-ea:>+10.6f}")


def print_rollout_comparison(name_a, ra_single, ra_ens, name_b, rb_single, rb_ens, lts=None):
    if lts is None:
        lts = sorted(set(ra_single.keys()) & set(rb_single.keys()))
    print("\n" + "=" * 110)
    print(f"  AUTOREGRESSIVE ROLLOUT EDM COMPARISON (ensemble mean metrics)")
    print("=" * 110)
    print(f"  {'LT':>3s} | {'MSE '+name_a[:12]:>14s} {'MSE '+name_b[:12]:>14s} {'Δ':>10s} | {'SSIM '+name_a[:12]:>14s} {'SSIM '+name_b[:12]:>14s} {'Δ':>10s}")
    print("-" * 110)
    for lt in lts:
        a_mse = ra_ens[lt]['mse'].mean()
        b_mse = rb_ens[lt]['mse'].mean()
        a_ssim = ra_ens[lt]['ssim'].mean()
        b_ssim = rb_ens[lt]['ssim'].mean()
        print(f"  {lt:>3d} | {a_mse:>14.6f} {b_mse:>14.6f} {b_mse-a_mse:>+10.6f} | {a_ssim:>14.6f} {b_ssim:>14.6f} {b_ssim-a_ssim:>+10.6f}")


# ============================================================
# Main
# ============================================================

SAMPLING_PARAMS = dict(
    num_steps=36,
    sigma_min=0.002,
    sigma_max=140,
    rho=4,
    S_churn=7.2,
    S_min=0,
    S_max=float('inf'),
    S_noise=1,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-a', type=str,
                        default='/home/group1/26fall_aiclass/ly/cira-diff/outputs/edm_full/',
                        help='EXP-011 reproduced EDM (diffusers format)')
    parser.add_argument('--ckpt-a-format', type=str, default='diffusers', choices=['diffusers', 'pth'])
    parser.add_argument('--ckpt-b', type=str,
                        default='/data1/satcast/edm_plain_diffusion/edm_plain_diffusion/checkpoint.pth',
                        help='Official EDM (pth format)')
    parser.add_argument('--ckpt-b-format', type=str, default='pth', choices=['diffusers', 'pth'])
    parser.add_argument('--name-a', type=str, default='EXP-011 EDM (287ep)', help='Display name A')
    parser.add_argument('--name-b', type=str, default='Official EDM (1000ep)', help='Display name B')
    parser.add_argument('--test-zarr', type=str, default='/data1/satcast/edm_GOES_ch13_test_dataset.zarr')
    parser.add_argument('--outdir', type=str,
                        default='/home/group1/26fall_aiclass/ly/cira-diff/outputs/eval_edm_comparison')
    parser.add_argument('--ens-size', type=int, default=10)
    parser.add_argument('--max-single-step', type=int, default=128,
                        help='Samples for single-step eval (128=smoke, 1024=full)')
    parser.add_argument('--max-rollout', type=int, default=128,
                        help='Samples for rollout eval')
    parser.add_argument('--base-seed', type=int, default=42)
    parser.add_argument('--skip-rollout', action='store_true')
    parser.add_argument('--skip-single-step', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model-filter', type=str, default='both', choices=['a', 'b', 'both'],
                        help='Run only checkpoint a, b, or both')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Canonical sampling: {SAMPLING_PARAMS}")
    print(f"Ensemble size: {args.ens_size}")
    print(f"Model filter: {args.model_filter}")

    print(f"\nLoading test set: {args.test_zarr}")
    dataset = TestZarrDataset(args.test_zarr)
    print(f"  Test samples: {len(dataset)}")

    model_a = model_b = None
    sm_a = em_a = sm_b = em_b = None
    ra_single = ra_ens = rb_single = rb_ens = None

    if args.model_filter in ('a', 'both'):
        print(f"\n{'='*60}")
        print(f"Loading Model A: {args.name_a}")
        print(f"  path: {args.ckpt_a}")
        print(f"  format: {args.ckpt_a_format}")
        print(f"{'='*60}")
        model_a = load_edm_model(args.ckpt_a, args.ckpt_a_format, device=device)

    if args.model_filter in ('b', 'both'):
        print(f"\n{'='*60}")
        print(f"Loading Model B: {args.name_b}")
        print(f"  path: {args.ckpt_b}")
        print(f"  format: {args.ckpt_b_format}")
        print(f"{'='*60}")
        model_b = load_edm_model(args.ckpt_b, args.ckpt_b_format, device=device)

    torch.cuda.empty_cache()
    gc.collect()

    # ========== Phase 1: Single-step ==========
    if not args.skip_single_step:
        print("\n" + "#" * 60)
        print("# Phase 1: Single-step EDM ensemble evaluation")
        print("#" * 60)

        if model_a is not None:
            sm_a, em_a, _ = evaluate_single_step_ensemble(
                model_a, dataset, device=device, ens_size=args.ens_size,
                max_samples=args.max_single_step, base_seed=args.base_seed,
                **SAMPLING_PARAMS
            )
            torch.cuda.empty_cache()
            gc.collect()

        if model_b is not None:
            sm_b, em_b, _ = evaluate_single_step_ensemble(
                model_b, dataset, device=device, ens_size=args.ens_size,
                max_samples=args.max_single_step, base_seed=args.base_seed,
                **SAMPLING_PARAMS
            )

        if model_a is not None and model_b is not None:
            print_single_step_comparison(args.name_a, sm_a, em_a, args.name_b, sm_b, em_b)

        if sm_a is not None:
            np.savez(os.path.join(args.outdir, 'single_step_a_single.npz'), **{k: sm_a[k] for k in sm_a})
            np.savez(os.path.join(args.outdir, 'single_step_a_ens.npz'), **{k: em_a[k] for k in em_a})
        if sm_b is not None:
            np.savez(os.path.join(args.outdir, 'single_step_b_single.npz'), **{k: sm_b[k] for k in sm_b})
            np.savez(os.path.join(args.outdir, 'single_step_b_ens.npz'), **{k: em_b[k] for k in em_b})

        torch.cuda.empty_cache()
        gc.collect()

    # ========== Phase 2: Rollout ==========
    if not args.skip_rollout:
        print("\n" + "#" * 60)
        print("# Phase 2: 18-step autoregressive rollout EDM ensemble")
        print("#" * 60)

        if model_a is not None:
            ra_single, ra_ens = evaluate_rollout_ensemble(
                model_a, dataset, num_rollout_steps=18, ens_size=args.ens_size,
                device=device, max_samples=args.max_rollout,
                base_seed=args.base_seed, **SAMPLING_PARAMS
            )
            torch.cuda.empty_cache()
            gc.collect()

        if model_b is not None:
            rb_single, rb_ens = evaluate_rollout_ensemble(
                model_b, dataset, num_rollout_steps=18, ens_size=args.ens_size,
                device=device, max_samples=args.max_rollout,
                base_seed=args.base_seed, **SAMPLING_PARAMS
            )

        if model_a is not None and model_b is not None:
            print_rollout_comparison(args.name_a, ra_single, ra_ens,
                                     args.name_b, rb_single, rb_ens)

        def save_rollout(path, rd_single, rd_ens):
            out = {}
            for lt in sorted(rd_single.keys()):
                for k in rd_single[lt]:
                    out[f'lt{lt}_single_{k}'] = [float(x) for x in rd_single[lt][k]]
                    out[f'lt{lt}_ens_{k}'] = [float(x) for x in rd_ens[lt][k]]
            with open(path, 'w') as f:
                json.dump(out, f, indent=2)

        if ra_single is not None:
            save_rollout(os.path.join(args.outdir, 'rollout_a.json'), ra_single, ra_ens)
        if rb_single is not None:
            save_rollout(os.path.join(args.outdir, 'rollout_b.json'), rb_single, rb_ens)

    # ========== Summary JSON ==========
    summary = {
        'ckpt_a': args.ckpt_a, 'ckpt_a_format': args.ckpt_a_format, 'name_a': args.name_a,
        'ckpt_b': args.ckpt_b, 'ckpt_b_format': args.ckpt_b_format, 'name_b': args.name_b,
        'test_zarr': args.test_zarr,
        'num_test_samples': len(dataset),
        'ens_size': args.ens_size,
        'sampling_params': SAMPLING_PARAMS,
        'base_seed': args.base_seed,
    }
    with open(os.path.join(args.outdir, 'eval_config.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nConfig saved to {args.outdir}/eval_config.json")
    print("Done.")


if __name__ == '__main__':
    main()