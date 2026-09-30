import os, sys, json, gc, argparse
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def ssim_pytorch(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """Minimal SSIM, operates on [B, C, H, W] tensors. Input range assumed in [0, 1]."""
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    channels = img1.shape[1]
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
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


def compute_metrics(pred, target):
    """pred, target: numpy arrays of shape (H, W) or (1, H, W) or (H, W). Returns dict."""
    if pred.ndim == 2:
        pred = pred[None, None, ...]
    elif pred.ndim == 3:
        pred = pred[None, ...]
    if target.ndim == 2:
        target = target[None, None, ...]
    elif target.ndim == 3:
        target = target[None, ...]
    pt = torch.from_numpy(pred.astype(np.float32))
    tt = torch.from_numpy(target.astype(np.float32))
    mse = F.mse_loss(pt, tt).item()
    mae = F.l1_loss(pt, tt).item()
    ssim = ssim_pytorch(pt, tt).item()
    psnr = psnr_pytorch(pt, tt).item()
    return {'mse': mse, 'mae': mae, 'ssim': ssim, 'psnr': psnr}


class TestZarrDataset(Dataset):
    """
    Test/Validation zarr with output_images shape (N, 18, 256, 256).
    input_images shape (N, 2, 256, 256): frames (t-1, t).
    __getitem__ returns (inputs [2, H, W], targets [18, H, W]).
    """
    def __init__(self, zarr_store):
        self.data = zarr.open(zarr_store, mode='r')
        self.input_images = self.data['input_images']
        self.output_images = self.data['output_images']
        self.num_samples = self.input_images.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        inputs = torch.from_numpy(self.input_images[idx].astype(np.float32))
        targets = torch.from_numpy(self.output_images[idx].astype(np.float32))
        return inputs, targets


def rollout_autoregressive(model, init_inputs, num_steps=18, device='cuda'):
    """
    init_inputs: [2, H, W] tensor (frames t-1, t).
    Returns preds: list of [1, H, W], length num_steps (predicted t+1 ... t+num_steps).
    """
    model.eval()
    preds = []
    current = init_inputs.to(device).unsqueeze(0)  # [1, 2, H, W]
    with torch.no_grad():
        for step in range(num_steps):
            pred = model(current, torch.zeros(1, device=device)).sample  # [1, 1, H, W]
            preds.append(pred.squeeze(0).cpu())
            current = torch.cat([current[:, 1:], pred], dim=1)
    return preds


def evaluate_single_step(model, dataset, device='cuda', verbose=True):
    """Teacher-forced single-step: use output_images[:, 0:1] as ground-truth t+1."""
    all_metrics = {'mse': [], 'mae': [], 'ssim': [], 'psnr': []}
    model.eval()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)  # [B, 2, H, W]
            target_t1 = targets[:, 0:1, :, :].to(device)  # [B, 1, H, W]
            pred = model(inputs, torch.zeros(inputs.shape[0], device=device)).sample
            mse = F.mse_loss(pred.float(), target_t1.float(), reduction='none').mean(dim=(1, 2, 3)).cpu().numpy()
            mae = F.l1_loss(pred.float(), target_t1.float(), reduction='none').mean(dim=(1, 2, 3)).cpu().numpy()
            ssim_vals = ssim_pytorch(pred.float(), target_t1.float()).cpu().numpy()
            psnr_vals = psnr_pytorch(pred.float(), target_t1.float()).cpu().numpy()
            all_metrics['mse'].extend(mse.tolist())
            all_metrics['mae'].extend(mae.tolist())
            all_metrics['ssim'].extend(ssim_vals.tolist())
            all_metrics['psnr'].extend(psnr_vals.tolist())
    for k in all_metrics:
        all_metrics[k] = np.array(all_metrics[k])
    if verbose:
        print(f"  single-step teacher-forced (n={len(all_metrics['mse'])})")
        for k in ['mse', 'mae', 'ssim', 'psnr']:
            print(f"    {k.upper():6s}  mean={all_metrics[k].mean():.6f}  std={all_metrics[k].std():.6f}  med={np.median(all_metrics[k]):.6f}")
    return all_metrics


def evaluate_rollout(model, dataset, num_steps=18, device='cuda', max_samples=None, verbose=True):
    """
    Autoregressive rollout.
    Returns per-lead-time metrics: dict lead_time (1..num_steps) -> dict of arrays.
    """
    per_lt = {lt: {'mse': [], 'mae': [], 'ssim': [], 'psnr': []} for lt in range(1, num_steps + 1)}
    model.eval()
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    if verbose:
        print(f"  autoregressive rollout (n={n}, steps={num_steps}) ...")
    for idx in range(n):
        inputs, targets = dataset[idx]
        preds = rollout_autoregressive(model, inputs, num_steps=num_steps, device=device)
        for lt in range(num_steps):
            pred_np = preds[lt].numpy()
            target_np = targets[lt, None].numpy()
            m = compute_metrics(pred_np, target_np)
            per_lt[lt + 1]['mse'].append(m['mse'])
            per_lt[lt + 1]['mae'].append(m['mae'])
            per_lt[lt + 1]['ssim'].append(m['ssim'])
            per_lt[lt + 1]['psnr'].append(m['psnr'])
    for lt in per_lt:
        for k in per_lt[lt]:
            per_lt[lt][k] = np.array(per_lt[lt][k])
    if verbose:
        print(f"  rollout done. Mean per lead time:")
        header = f"  {'LT':>3s}  {'MSE':>10s}  {'MAE':>10s}  {'SSIM':>10s}  {'PSNR':>10s}"
        print(header)
        for lt in [1, 3, 6, 9, 12, 15, 18]:
            if lt in per_lt:
                d = per_lt[lt]
                print(f"  {lt:>3d}  {d['mse'].mean():>10.6f}  {d['mae'].mean():>10.6f}  {d['ssim'].mean():>10.6f}  {d['psnr'].mean():>10.4f}")
    return per_lt


def print_comparison_table(name_a, metrics_a, name_b, metrics_b):
    print("\n" + "=" * 65)
    print(f"  Single-step teacher-forced comparison on TEST SET")
    print("=" * 65)
    print(f"  {'Metric':>6s}  {name_a:>30s}  {name_b:>30s}  {'Δ (B-A)':>10s}")
    print("-" * 65)
    for k in ['mse', 'mae', 'ssim', 'psnr']:
        a_mean = metrics_a[k].mean()
        b_mean = metrics_b[k].mean()
        delta = b_mean - a_mean
        print(f"  {k.upper():>6s}  {a_mean:>30.6f}  {b_mean:>30.6f}  {delta:>+10.6f}")


def print_rollout_comparison(name_a, roll_a, name_b, roll_b, lts=None):
    if lts is None:
        lts = sorted(set(roll_a.keys()) & set(roll_b.keys()))
    print("\n" + "=" * 95)
    print(f"  Autoregressive Rollout Comparison on TEST SET")
    print("=" * 95)
    header = f"  {'LT':>3s} | {'MSE ' + name_a[:12]:>14s} {'MSE ' + name_b[:12]:>14s} {'Δ':>10s} | {'SSIM ' + name_a[:12]:>14s} {'SSIM ' + name_b[:12]:>14s} {'Δ':>10s}"
    print(header)
    print("-" * 95)
    for lt in lts:
        a = roll_a[lt]
        b = roll_b[lt]
        mse_a, mse_b = a['mse'].mean(), b['mse'].mean()
        ssim_a, ssim_b = a['ssim'].mean(), b['ssim'].mean()
        print(f"  {lt:>3d} | {mse_a:>14.6f} {mse_b:>14.6f} {mse_b - mse_a:>+10.6f} | {ssim_a:>14.6f} {ssim_b:>14.6f} {ssim_b - ssim_a:>+10.6f}")


def plot_rollout_curves(name_a, roll_a, name_b, roll_b, outdir):
    lts = sorted(set(roll_a.keys()) & set(roll_b.keys()))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, metric in zip(axes.flat, ['mse', 'mae', 'ssim', 'psnr']):
        a_vals = [roll_a[lt][metric].mean() for lt in lts]
        b_vals = [roll_b[lt][metric].mean() for lt in lts]
        ax.plot(lts, a_vals, marker='o', label=name_a)
        ax.plot(lts, b_vals, marker='s', label=name_b)
        ax.set_xlabel('Lead Time (step)')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} vs Lead Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rollout_metrics_curves.png'), dpi=150)
    plt.close()


def plot_visual_comparison(model_a, model_b, dataset, outdir, device='cuda', num_samples=4):
    model_a.eval()
    model_b.eval()
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    fig, axes = plt.subplots(num_samples, 8, figsize=(24, 3.5 * num_samples))
    cmapper = plt.get_cmap('magma')
    for row, idx in enumerate(indices):
        inputs, targets = dataset[idx]
        with torch.no_grad():
            pred_a = model_a(inputs.unsqueeze(0).to(device), torch.zeros(1, device=device)).sample.cpu().squeeze(0)
            pred_b = model_b(inputs.unsqueeze(0).to(device), torch.zeros(1, device=device)).sample.cpu().squeeze(0)
        target_t1 = targets[0:1]
        vmin = min(inputs.min(), target_t1.min(), pred_a.min(), pred_b.min())
        vmax = max(inputs.max(), target_t1.max(), pred_a.max(), pred_b.max())

        def show(ax, img, title):
            im = ax.imshow(img.squeeze().numpy(), cmap=cmapper, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
            return im

        show(axes[row, 0], inputs[0], 'Input t-1')
        show(axes[row, 1], inputs[1], 'Input t')
        show(axes[row, 2], pred_a, f'Model A pred')
        show(axes[row, 3], target_t1, 'Ground truth t+1')
        show(axes[row, 4], pred_b, f'Model B pred')
        show(axes[row, 5], target_t1, 'Ground truth t+1')
        err_a = np.abs(pred_a.numpy() - target_t1.numpy()).squeeze()
        err_b = np.abs(pred_b.numpy() - target_t1.numpy()).squeeze()
        im_a = axes[row, 6].imshow(err_a, cmap='inferno', vmin=0, vmax=max(err_a.max(), err_b.max()))
        axes[row, 6].set_title(f'|err| A', fontsize=9); axes[row, 6].axis('off')
        im_b = axes[row, 7].imshow(err_b, cmap='inferno', vmin=0, vmax=max(err_a.max(), err_b.max()))
        axes[row, 7].set_title(f'|err| B', fontsize=9); axes[row, 7].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'visual_comparison_single_step.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-a', type=str,
                        default='/home/group1/26fall_aiclass/ly/cira-diff/outputs/vanilla_unet_full/best_internal_unet',
                        help='Path to model A (e.g. user-trained vanilla UNet)')
    parser.add_argument('--ckpt-b', type=str,
                        default='/data1/satcast/unet_vanilla/unet_vanilla',
                        help='Path to model B (e.g. paper open-source vanilla UNet)')
    parser.add_argument('--name-a', type=str, default='EXP-010 Trained', help='Display name for model A')
    parser.add_argument('--name-b', type=str, default='Paper OpenSource', help='Display name for model B')
    parser.add_argument('--test-zarr', type=str, default='/data1/satcast/edm_GOES_ch13_test_dataset.zarr')
    parser.add_argument('--outdir', type=str, default='/home/group1/26fall_aiclass/ly/cira-diff/outputs/eval_unet_comparison')
    parser.add_argument('--skip-rollout', action='store_true')
    parser.add_argument('--max-rollout-samples', type=int, default=128,
                        help='Subset size for rollout eval (full 1024 is ~30 min/model on 4090)')
    parser.add_argument('--single-step-only-rollout', action='store_true',
                        help='Use single-step teacher-forced for "rollout" (no autoregressive)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print(f"\nLoading dataset: {args.test_zarr}")
    dataset = TestZarrDataset(args.test_zarr)
    print(f"  Test samples: {len(dataset)}")

    print(f"\nLoading Model A: {args.name_a} from {args.ckpt_a}")
    model_a = UNet2DModel.from_pretrained(args.ckpt_a)
    model_a.to(device)
    model_a.eval()

    print(f"Loading Model B: {args.name_b} from {args.ckpt_b}")
    model_b = UNet2DModel.from_pretrained(args.ckpt_b)
    model_b.to(device)
    model_b.eval()

    torch.cuda.empty_cache()
    gc.collect()

    # ========== 1. Single-step teacher-forced ==========
    print("\n" + "#" * 60)
    print("# 1. Single-step teacher-forced evaluation")
    print("#" * 60)
    print(f"\n[{args.name_a}]")
    m_a = evaluate_single_step(model_a, dataset, device=device)
    torch.cuda.empty_cache()
    print(f"\n[{args.name_b}]")
    m_b = evaluate_single_step(model_b, dataset, device=device)

    print_comparison_table(args.name_a, m_a, args.name_b, m_b)

    # Save single-step metrics
    np.savez(os.path.join(args.outdir, 'single_step_metrics_a.npz'), **{k: m_a[k] for k in m_a})
    np.savez(os.path.join(args.outdir, 'single_step_metrics_b.npz'), **{k: m_b[k] for k in m_b})

    # ========== 2. Autoregressive rollout ==========
    if not args.skip_rollout:
        print("\n" + "#" * 60)
        print("# 2. Autoregressive rollout evaluation (18 steps)")
        print("#" * 60)
        if args.single_step_only_rollout:
            print("  [MODE] single-step teacher-forced rollout (not autoregressive)")
            r_a = evaluate_rollout_teacherforced_multi(model_a, dataset, device=device, max_samples=args.max_rollout_samples)
            torch.cuda.empty_cache()
            r_b = evaluate_rollout_teacherforced_multi(model_b, dataset, device=device, max_samples=args.max_rollout_samples)
        else:
            print(f"  [MODE] autoregressive rollout (n={args.max_rollout_samples} subset)")
            r_a = evaluate_rollout(model_a, dataset, num_steps=18, device=device, max_samples=args.max_rollout_samples)
            torch.cuda.empty_cache()
            r_b = evaluate_rollout(model_b, dataset, num_steps=18, device=device, max_samples=args.max_rollout_samples)

        print_rollout_comparison(args.name_a, r_a, args.name_b, r_b)
        plot_rollout_curves(args.name_a, r_a, args.name_b, r_b, args.outdir)

        r_a_save = {f'lt{k}': {kk: r_a[k][kk] for kk in r_a[k]} for k in r_a}
        r_b_save = {f'lt{k}': {kk: r_b[k][kk] for kk in r_b[k]} for k in r_b}
        with open(os.path.join(args.outdir, 'rollout_metrics_a.json'), 'w') as f:
            json.dump({f'lt{k}': {kk: [float(x) for x in r_a[k][kk]] for kk in r_a[k]} for k in r_a}, f, indent=2)
        with open(os.path.join(args.outdir, 'rollout_metrics_b.json'), 'w') as f:
            json.dump({f'lt{k}': {kk: [float(x) for x in r_b[k][kk]] for kk in r_b[k]} for k in r_b}, f, indent=2)

    # ========== 3. Visual comparison ==========
    print("\n" + "#" * 60)
    print("# 3. Visual comparison (single-step)")
    print("#" * 60)
    plot_visual_comparison(model_a, model_b, dataset, args.outdir, device=device, num_samples=4)
    print(f"  saved visual_comparison_single_step.png")

    # ========== 4. Summary JSON ==========
    summary = {
        'ckpt_a': args.ckpt_a,
        'ckpt_b': args.ckpt_b,
        'name_a': args.name_a,
        'name_b': args.name_b,
        'test_zarr': args.test_zarr,
        'num_test_samples': len(dataset),
        'single_step': {
            'model_a': {k: {'mean': float(m_a[k].mean()), 'std': float(m_a[k].std()), 'median': float(np.median(m_a[k]))} for k in m_a},
            'model_b': {k: {'mean': float(m_b[k].mean()), 'std': float(m_b[k].std()), 'median': float(np.median(m_b[k]))} for k in m_b},
        },
    }
    with open(os.path.join(args.outdir, 'comparison_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {args.outdir}/comparison_summary.json")
    print("Done.")


def evaluate_rollout_teacherforced_multi(model, dataset, device='cuda', max_samples=None):
    """Alternative: teacher-forced prediction for each lead time independently."""
    per_lt = {lt: {'mse': [], 'mae': [], 'ssim': [], 'psnr': []} for lt in range(1, 19)}
    model.eval()
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    collected = {lt: {'preds': [], 'targets': []} for lt in range(1, 19)}
    count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            if count >= n: break
            bs = inputs.shape[0]
            inputs = inputs.to(device)
            for lt in range(1, 19):
                tgt = targets[:, lt - 1:lt, :, :].to(device)
                pred = model(inputs, torch.zeros(bs, device=device)).sample
                collected[lt]['preds'].append(pred.cpu())
                collected[lt]['targets'].append(tgt.cpu())
            count += bs
    for lt in range(1, 19):
        preds = torch.cat(collected[lt]['preds'], dim=0)[:n].float()
        tgts = torch.cat(collected[lt]['targets'], dim=0)[:n].float()
        mse = F.mse_loss(preds, tgts, reduction='none').mean(dim=(1, 2, 3)).numpy()
        mae = F.l1_loss(preds, tgts, reduction='none').mean(dim=(1, 2, 3)).numpy()
        ssim = ssim_pytorch(preds, tgts).numpy()
        psnr = psnr_pytorch(preds, tgts).numpy()
        per_lt[lt]['mse'] = mse
        per_lt[lt]['mae'] = mae
        per_lt[lt]['ssim'] = ssim
        per_lt[lt]['psnr'] = psnr
    print(f"  teacher-forced multi-step (n={n})")
    print(f"  {'LT':>3s}  {'MSE':>10s}  {'MAE':>10s}  {'SSIM':>10s}  {'PSNR':>10s}")
    for lt in [1, 3, 6, 9, 12, 15, 18]:
        d = per_lt[lt]
        print(f"  {lt:>3d}  {d['mse'].mean():>10.6f}  {d['mae'].mean():>10.6f}  {d['ssim'].mean():>10.6f}  {d['psnr'].mean():>10.4f}")
    return per_lt


if __name__ == '__main__':
    main()