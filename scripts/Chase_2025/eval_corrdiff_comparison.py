"""
CorrDiff Evaluation: Official vs EXP-012 Reproduced CorrDiff
============================================================
Evaluates CorrDiff as a forecast correction diffusion model.
CorrDiff predicts residuals (forecast errors), so final forecast = residual + UNet forecast (condition ch2).

Key differences from EDM eval:
- UNet in_channels = 4 (1 noisy residual + 3 condition)
- sigma_data = 0.5 (same as EDM)
- Output is residual, so metrics must be computed after reconstruction
- Condition has 3 channels; ch2 is the UNet forecast base
"""

import os, sys, json, gc, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel
from tqdm import tqdm


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


class CorrDiffValDataset(Dataset):
    def __init__(self, zarr_store):
        self.data = zarr.open(zarr_store, mode='r')
        self.input_images = self.data['input_images']
        self.output_images = self.data['output_images']
        self.num_samples = self.input_images.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        inputs = torch.from_numpy(self.input_images[idx].astype(np.float32))
        residual_target = torch.from_numpy(self.output_images[idx].astype(np.float32))
        unet_forecast = inputs[2:3].clone()
        return inputs, residual_target, unet_forecast


class CorrDiffRolloutDataset(Dataset):
    """
    For autoregressive rollout: uses EDM test zarr but builds CorrDiff condition on-the-fly.
    EDM test zarr: input=(N, 2, 256, 256) [t-1 UNet forecast, t UNet forecast]
                  output=(N, 18, 256, 256) truth for 18 steps
    CorrDiff condition (3ch): ch0=UNet t-1, ch1=UNet t, ch2=UNet t (UNet forecast base for residual reconstruction)
    """
    def __init__(self, zarr_store):
        self.data = zarr.open(zarr_store, mode='r')
        self.input_images = self.data['input_images']
        self.output_images = self.data['output_images']
        self.num_samples = self.input_images.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        edm_input = torch.from_numpy(self.input_images[idx].astype(np.float32))
        truth_targets = torch.from_numpy(self.output_images[idx].astype(np.float32))
        corrdiff_cond = torch.stack([edm_input[0], edm_input[1], edm_input[1]], dim=0)
        return corrdiff_cond, truth_targets


def build_corrdiff_unet():
    return UNet2DModel(
        sample_size=256,
        in_channels=4,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D",
                          "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                        "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


def load_corrdiff_model(ckpt_path, ckpt_format, device='cuda'):
    if ckpt_format == 'diffusers':
        base = UNet2DModel.from_pretrained(ckpt_path)
        assert base.config.in_channels == 4, f"Expected in_channels=4, got {base.config.in_channels}"
    elif ckpt_format == 'pth':
        base = build_corrdiff_unet()
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
            epoch = ckpt.get('epoch', '?')
            step = ckpt.get('step', '?')
            print(f"  Loaded checkpoint: epoch={epoch}, step={step}")
        else:
            state = ckpt
            print(f"  Loaded raw state_dict (no epoch/step info)")
        missing, unexpected = base.load_state_dict(state, strict=True)
        assert len(missing) == 0 and len(unexpected) == 0, f"Load failed: missing={missing}, unexpected={unexpected}"
    else:
        raise ValueError(f"Unknown ckpt_format: {ckpt_format}")

    wrapped = EDMPrecond(generation_channels=1, model=base, use_fp16=False,
                         sigma_min=0, sigma_max=float('inf'), sigma_data=0.5)
    wrapped.to(device)
    wrapped.eval()
    return wrapped


def evaluate_single_step_residual(model_cd, dataset, device='cuda',
                                   ens_size=10, num_steps=36,
                                   sigma_min=0.002, sigma_max=140, rho=4,
                                   S_churn=7.2, S_min=0, S_max=float('inf'), S_noise=1,
                                   max_samples=None, base_seed=0):
    """
    Single-step CorrDiff evaluation.
    Outputs residual predictions; full forecast = residual + unet_forecast.
    Reports metrics in BOTH residual space and full forecast space.
    """
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    results = {
        'residual_single': {'mse': [], 'mae': [], 'ssim': [], 'psnr': []},
        'residual_ens': {'mse': [], 'mae': [], 'ssim': [], 'psnr': []},
        'forecast_single': {'mse': [], 'mae': [], 'ssim': [], 'psnr': []},
        'forecast_ens': {'mse': [], 'mae': [], 'ssim': [], 'psnr': []},
    }

    model_cd.eval()

    for idx in tqdm(range(n), desc=f"Single-step CorrDiff (ens={ens_size}, n={n})", mininterval=5):
        inputs, residual_target, unet_forecast = dataset[idx]
        cond = inputs.unsqueeze(0).to(device)
        residual_target = residual_target.to(device)
        unet_forecast = unet_forecast.to(device)

        seeds = [base_seed + idx * ens_size + m for m in range(ens_size)]
        rnd = StackedRandomGenerator(device, seeds)
        latents = rnd.randn([ens_size, 1, 256, 256], device=device)
        cond_repeated = cond.repeat(ens_size, 1, 1, 1)

        with torch.no_grad():
            residual_preds = edm_sampler(model_cd, latents, cond_repeated,
                                num_steps=num_steps, sigma_min=sigma_min,
                                sigma_max=sigma_max, rho=rho,
                                S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise)

        target_rep = residual_target.unsqueeze(0).repeat(ens_size, 1, 1, 1)

        mse_ind = F.mse_loss(residual_preds.float(), target_rep.float(), reduction='none').mean(dim=(1, 2, 3)).cpu().numpy()
        mae_ind = F.l1_loss(residual_preds.float(), target_rep.float(), reduction='none').mean(dim=(1, 2, 3)).cpu().numpy()
        ssim_ind = ssim_pytorch(residual_preds.float(), target_rep.float()).cpu().numpy()
        psnr_ind = psnr_pytorch(residual_preds.float(), target_rep.float()).cpu().numpy()

        results['residual_single']['mse'].append(mse_ind.mean())
        results['residual_single']['mae'].append(mae_ind.mean())
        results['residual_single']['ssim'].append(ssim_ind.mean())
        results['residual_single']['psnr'].append(psnr_ind.mean())

        residual_ens = residual_preds.mean(dim=0, keepdim=True)
        mse_ens = F.mse_loss(residual_ens.float(), residual_target.unsqueeze(0).float()).item()
        mae_ens = F.l1_loss(residual_ens.float(), residual_target.unsqueeze(0).float()).item()
        ssim_ens = ssim_pytorch(residual_ens.float(), residual_target.unsqueeze(0).float()).item()
        psnr_ens = psnr_pytorch(residual_ens.float(), residual_target.unsqueeze(0).float()).item()

        results['residual_ens']['mse'].append(mse_ens)
        results['residual_ens']['mae'].append(mae_ens)
        results['residual_ens']['ssim'].append(ssim_ens)
        results['residual_ens']['psnr'].append(psnr_ens)

        forecast_preds = residual_preds + unet_forecast.unsqueeze(0).repeat(ens_size, 1, 1, 1)
        forecast_target = residual_target + unet_forecast

        forecast_single_mse = F.mse_loss(forecast_preds.float(), forecast_target.unsqueeze(0).float(), reduction='none').mean(dim=(1, 2, 3)).mean().item()
        forecast_single_mae = F.l1_loss(forecast_preds.float(), forecast_target.unsqueeze(0).float(), reduction='none').mean(dim=(1, 2, 3)).mean().item()
        forecast_single_ssim = ssim_pytorch(forecast_preds.float(), forecast_target.unsqueeze(0).float()).mean().item()
        forecast_single_psnr = psnr_pytorch(forecast_preds.float(), forecast_target.unsqueeze(0).float()).mean().item()

        results['forecast_single']['mse'].append(forecast_single_mse)
        results['forecast_single']['mae'].append(forecast_single_mae)
        results['forecast_single']['ssim'].append(forecast_single_ssim)
        results['forecast_single']['psnr'].append(forecast_single_psnr)

        forecast_ens = residual_ens + unet_forecast.unsqueeze(0)
        results['forecast_ens']['mse'].append(F.mse_loss(forecast_ens.float(), forecast_target.unsqueeze(0).float()).item())
        results['forecast_ens']['mae'].append(F.l1_loss(forecast_ens.float(), forecast_target.unsqueeze(0).float()).item())
        results['forecast_ens']['ssim'].append(ssim_pytorch(forecast_ens.float(), forecast_target.unsqueeze(0).float()).item())
        results['forecast_ens']['psnr'].append(psnr_pytorch(forecast_ens.float(), forecast_target.unsqueeze(0).float()).item())

    for k in results:
        for metric in results[k]:
            results[k][metric] = np.array(results[k][metric])

    return results


def evaluate_unet_baseline(dataset, max_samples=None):
    """Evaluate the raw UNet forecast (no correction) as baseline."""
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    forecast_mse = []
    forecast_mae = []
    forecast_ssim = []

    for idx in tqdm(range(n), desc="UNet baseline (no correction)", mininterval=5):
        inputs, residual_target, unet_forecast = dataset[idx]
        truth = residual_target + unet_forecast
        unet_err = unet_forecast - truth
        forecast_mse.append(F.mse_loss(unet_forecast.unsqueeze(0), truth.unsqueeze(0)).item())
        forecast_mae.append(F.l1_loss(unet_forecast.unsqueeze(0), truth.unsqueeze(0)).item())
        forecast_ssim.append(ssim_pytorch(unet_forecast.unsqueeze(0), truth.unsqueeze(0)).item())

    return {
        'forecast_mse': np.array(forecast_mse),
        'forecast_mae': np.array(forecast_mae),
        'forecast_ssim': np.array(forecast_ssim),
    }


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
                        default='/home/group1/26fall_aiclass/ly/cira-diff/outputs/corrdiff_full/',
                        help='EXP-012 reproduced CorrDiff (diffusers format)')
    parser.add_argument('--ckpt-a-format', type=str, default='diffusers', choices=['diffusers', 'pth'])
    parser.add_argument('--ckpt-b', type=str,
                        default='/data1/satcast/edm_corrdiff/edm_corrdiff/checkpoint.pth',
                        help='Official CorrDiff (pth format)')
    parser.add_argument('--ckpt-b-format', type=str, default='pth', choices=['diffusers', 'pth'])
    parser.add_argument('--name-a', type=str, default='EXP-012 CorrDiff', help='Display name A')
    parser.add_argument('--name-b', type=str, default='Official CorrDiff (1000ep)', help='Display name B')
    parser.add_argument('--val-zarr', type=str,
                        default='/data1/satcast/edm_GOES_ch13_val_dataset_CorrDiff.zarr',
                        help='CorrDiff val dataset (for single-step residual eval)')
    parser.add_argument('--outdir', type=str,
                        default='/home/group1/26fall_aiclass/ly/cira-diff/outputs/eval_corrdiff_comparison')
    parser.add_argument('--ens-size', type=int, default=10)
    parser.add_argument('--max-samples', type=int, default=256,
                        help='Samples for single-step eval (256=smoke, full val=1779)')
    parser.add_argument('--base-seed', type=int, default=42)
    parser.add_argument('--model-filter', type=str, default='both', choices=['a', 'b', 'both'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Canonical sampling: {SAMPLING_PARAMS}")
    print(f"Ensemble size: {args.ens_size}")

    print(f"\nLoading CorrDiff val set: {args.val_zarr}")
    dataset = CorrDiffValDataset(args.val_zarr)
    print(f"  Val samples: {len(dataset)}")

    print("\nRunning UNet baseline (no correction)...")
    baseline = evaluate_unet_baseline(dataset, max_samples=args.max_samples)
    print(f"  UNet MSE={baseline['forecast_mse'].mean():.6f}, "
          f"MAE={baseline['forecast_mae'].mean():.6f}, "
          f"SSIM={baseline['forecast_ssim'].mean():.6f}")

    model_a = model_b = None
    results_a = results_b = None

    if args.model_filter in ('a', 'both'):
        print(f"\n{'='*60}")
        print(f"Loading Model A: {args.name_a}")
        print(f"  path: {args.ckpt_a}")
        print(f"  format: {args.ckpt_a_format}")
        print(f"{'='*60}")
        if os.path.exists(args.ckpt_a):
            model_a = load_corrdiff_model(args.ckpt_a, args.ckpt_a_format, device=device)
        else:
            print("  SKIP (path does not exist yet - training not done)")

    if args.model_filter in ('b', 'both'):
        print(f"\n{'='*60}")
        print(f"Loading Model B: {args.name_b}")
        print(f"  path: {args.ckpt_b}")
        print(f"  format: {args.ckpt_b_format}")
        print(f"{'='*60}")
        if os.path.exists(args.ckpt_b):
            model_b = load_corrdiff_model(args.ckpt_b, args.ckpt_b_format, device=device)
        else:
            print("  SKIP (path does not exist)")

    torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "#" * 60)
    print("# Single-step CorrDiff evaluation")
    print("#" * 60)

    if model_a is not None:
        print(f"\n--- Evaluating {args.name_a} ---")
        results_a = evaluate_single_step_residual(
            model_a, dataset, device=device, ens_size=args.ens_size,
            max_samples=args.max_samples, base_seed=args.base_seed,
            **SAMPLING_PARAMS
        )
        torch.cuda.empty_cache()
        gc.collect()

    if model_b is not None:
        print(f"\n--- Evaluating {args.name_b} ---")
        results_b = evaluate_single_step_residual(
            model_b, dataset, device=device, ens_size=args.ens_size,
            max_samples=args.max_samples, base_seed=args.base_seed,
            **SAMPLING_PARAMS
        )

    # ============================================================
    # Print comparison
    # ============================================================
    print("\n" + "=" * 100)
    print("  CORRDIFF SINGLE-STEP COMPARISON (forecast space, ensemble mean)")
    print("=" * 100)
    names = []
    res_list = []
    if model_a is not None:
        names.append(args.name_a)
        res_list.append(results_a)
    if model_b is not None:
        names.append(args.name_b)
        res_list.append(results_b)

    print(f"\n  {'Model':>30s} | {'Forecast MSE':>12s} {'MAE':>12s} {'SSIM':>12s} {'PSNR':>12s} | {'Residual MSE':>12s}")
    print("  " + "-" * 100)
    print(f"  {'UNet baseline':>30s} | "
          f"{baseline['forecast_mse'].mean():>12.6f} "
          f"{baseline['forecast_mae'].mean():>12.6f} "
          f"{baseline['forecast_ssim'].mean():>12.6f} "
          f"{'---':>12s} | "
          f"{'---':>12s}")
    for name, res in zip(names, res_list):
        f_ens = res['forecast_ens']
        r_ens = res['residual_ens']
        print(f"  {name:>30s} | "
              f"{f_ens['mse'].mean():>12.6f} "
              f"{f_ens['mae'].mean():>12.6f} "
              f"{f_ens['ssim'].mean():>12.6f} "
              f"{f_ens['psnr'].mean():>12.2f} | "
              f"{r_ens['mse'].mean():>12.6f}")

    # ============================================================
    # Save results
    # ============================================================
    out_json = {
        'baseline': {
            'forecast_mse_mean': float(baseline['forecast_mse'].mean()),
            'forecast_mae_mean': float(baseline['forecast_mae'].mean()),
            'forecast_ssim_mean': float(baseline['forecast_ssim'].mean()),
        },
        'config': {
            'val_zarr': args.val_zarr,
            'num_val_samples': len(dataset),
            'ens_size': args.ens_size,
            'sampling_params': SAMPLING_PARAMS,
            'base_seed': args.base_seed,
        }
    }

    for label, res, name in [('A', results_a, args.name_a), ('B', results_b, args.name_b)]:
        if res is not None:
            out_json[f'model_{label}'] = {'name': name}
            for space in ['residual', 'forecast']:
                for agg in ['single', 'ens']:
                    key = f'{space}_{agg}'
                    out_json[f'model_{label}'][key] = {
                        k: float(res[key][k].mean()) for k in res[key]
                    }
                    out_json[f'model_{label}'][f'{key}_values'] = {
                        k: [float(v) for v in res[key][k]] for k in res[key]
                    }

    with open(os.path.join(args.outdir, 'eval_results.json'), 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f"\nResults saved to {args.outdir}/eval_results.json")

    print("\nDone.")


if __name__ == '__main__':
    main()