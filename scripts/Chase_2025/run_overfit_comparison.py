import torch
import torch.nn.functional as F
import zarr
import numpy as np
import os
import time
import gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import UNet2DModel
from safetensors.torch import load_file, save_file

torch.manual_seed(42)
np.random.seed(42)

OUTPUT_ROOT = "/home/group1/26fall_aiclass/ly/cira-diff/outputs/overfit_comparison"
os.makedirs(OUTPUT_ROOT, exist_ok=True)
SAMPLE_IDX = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 1000
LR = 1e-4


def load_sample(idx=0):
    raw = zarr.open("/data1/satcast/edm_GOES_ch13_train_dataset.zarr", mode="r")
    lat = zarr.open("/data1/satcast/edm_GOES_ch13_train_dataset_latent.zarr", mode="r")
    raw_inp = torch.tensor(raw["input_images"][idx], dtype=torch.float32, device=DEVICE)
    raw_out = torch.tensor(raw["output_images"][idx], dtype=torch.float32, device=DEVICE)
    lat_inp = torch.tensor(lat["input_images"][idx], dtype=torch.float32, device=DEVICE)
    lat_out = torch.tensor(lat["output_images"][idx], dtype=torch.float32, device=DEVICE)
    return raw_inp.unsqueeze(0), raw_out.unsqueeze(0), lat_inp.unsqueeze(0), lat_out.unsqueeze(0)


def make_unet(in_channels, out_channels, sample_size, block_out_channels=(128,128,256,256,512,512)):
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=2,
        block_out_channels=block_out_channels,
        down_block_types=("DownBlock2D","DownBlock2D","DownBlock2D","DownBlock2D","AttnDownBlock2D","DownBlock2D"),
        up_block_types=("UpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D"),
    )


def train_vanilla_unet(cond, target, epochs=EPOCHS):
    print(f"\n{'='*60}\n[Vanilla UNet] training, cond={cond.shape}, target={target.shape}\n{'='*60}")
    model = make_unet(cond.shape[1], target.shape[1], target.shape[-1]).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    losses = []
    t0 = time.time()
    for ep in range(epochs):
        optim.zero_grad()
        yhat = model(cond, torch.zeros(cond.shape[0], device=DEVICE), return_dict=False)[0]
        loss = F.mse_loss(yhat, target)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if ep % 200 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:4d}: loss={loss.item():.6e}")
    print(f"[Vanilla UNet] done in {time.time()-t0:.1f}s, final loss={losses[-1]:.6e}")
    return model, losses


class EDMPrecond(torch.nn.Module):
    def __init__(self, gen_ch, model, sigma_data=0.5):
        super().__init__()
        self.gen_ch = gen_ch
        self.model = model
        self.sigma_data = sigma_data

    def forward(self, x, sigma):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1,1,1,1)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4
        x_noisy = x[:, 0:self.gen_ch]
        x_cond = x[:, self.gen_ch:]
        model_in = torch.cat([x_noisy * c_in, x_cond], dim=1)
        F_x = self.model(model_in, c_noise.flatten(), return_dict=False)[0]
        return c_skip * x_noisy + c_out * F_x


def train_edm(cond, target, epochs=EPOCHS, sigma_data=0.5, P_mean=-1.2, P_std=1.2):
    print(f"\n{'='*60}\n[EDM] training, cond={cond.shape}, target={target.shape}\n{'='*60}")
    model = make_unet(cond.shape[1] + target.shape[1], target.shape[1], target.shape[-1])
    wrapped = EDMPrecond(target.shape[1], model, sigma_data).to(DEVICE)
    optim = torch.optim.AdamW(wrapped.model.parameters(), lr=LR)
    losses = []
    t0 = time.time()
    for ep in range(epochs):
        optim.zero_grad()
        rnd = torch.randn(1,1,1,1, device=DEVICE)
        sigma = (rnd * P_std + P_mean).exp()
        weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
        n = torch.randn_like(target) * sigma
        noisy = target + n
        model_in = torch.cat([noisy, cond], dim=1)
        denoised = wrapped(model_in, sigma)
        loss = (weight * (denoised - target)**2).mean()
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if ep % 200 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:4d}: loss={loss.item():.6e}")
    print(f"[EDM] done in {time.time()-t0:.1f}s, final loss={losses[-1]:.6e}")
    return wrapped, losses


def edm_sample(wrapped, cond, num_steps=20, sigma_min=0.002, sigma_max=80, rho=7):
    wrapped.eval()
    with torch.no_grad():
        sigma_min = max(sigma_min, 0)
        sigma_max = min(sigma_max, float("inf"))
        step_idx = torch.arange(num_steps, dtype=torch.float64, device=DEVICE)
        t_steps = (sigma_max**(1/rho) + step_idx/(num_steps-1)*(sigma_min**(1/rho)-sigma_max**(1/rho)))**rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        x_next = torch.randn(1, wrapped.gen_ch, cond.shape[-2], cond.shape[-1], device=DEVICE, dtype=torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            model_in = torch.cat([x_cur.to(torch.float32), cond], dim=1)
            denoised = wrapped(model_in, t_cur.to(torch.float32)).to(torch.float64)
            d_cur = (x_cur - denoised) / t_cur
            x_next = x_cur + (t_next - t_cur) * d_cur
            if i < num_steps - 1:
                model_in = torch.cat([x_next.to(torch.float32), cond], dim=1)
                denoised = wrapped(model_in, t_next.to(torch.float32)).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_cur + (t_next - t_cur) * (0.5*d_cur + 0.5*d_prime)
    wrapped.train()
    return x_next.to(torch.float32)


def main():
    raw_cond, raw_target, lat_cond, lat_target = load_sample(SAMPLE_IDX)
    print(f"Sample {SAMPLE_IDX}: raw_cond={raw_cond.shape}, raw_target={raw_target.shape}, lat_cond={lat_cond.shape}, lat_target={lat_target.shape}")

    # ── 1. Vanilla UNet ──
    vanilla_model, vanilla_losses = train_vanilla_unet(raw_cond, raw_target)
    with torch.no_grad():
        vanilla_pred = vanilla_model(raw_cond, torch.zeros(1, device=DEVICE), return_dict=False)[0]
    vanilla_mse = F.mse_loss(vanilla_pred, raw_target).item()
    vanilla_mae = F.l1_loss(vanilla_pred, raw_target).item()
    print(f"[Vanilla UNet] MSE={vanilla_mse:.6f}, MAE={vanilla_mae:.6f}")
    del vanilla_model; gc.collect(); torch.cuda.empty_cache()

    # ── 2. EDM ──
    edm_wrapped, edm_losses = train_edm(raw_cond, raw_target)
    edm_pred = edm_sample(edm_wrapped, raw_cond, num_steps=20)
    edm_mse = F.mse_loss(edm_pred, raw_target).item()
    edm_mae = F.l1_loss(edm_pred, raw_target).item()
    print(f"[EDM] MSE={edm_mse:.6f}, MAE={edm_mae:.6f}")
    del edm_wrapped; gc.collect(); torch.cuda.empty_cache()

    # ── 3. LDM (latent 64x64) ──
    ldm_wrapped, ldm_losses = train_edm(lat_cond, lat_target)
    ldm_pred = edm_sample(ldm_wrapped, lat_cond, num_steps=20)
    ldm_mse = F.mse_loss(ldm_pred, lat_target).item()
    ldm_mae = F.l1_loss(ldm_pred, lat_target).item()
    print(f"[LDM] MSE={ldm_mse:.6f}, MAE={ldm_mae:.6f} (in latent space, 4ch)")
    ldm_pred_ch0 = ldm_pred[:, 0:1]
    lat_target_ch0 = lat_target[:, 0:1]
    ldm_pred_up = F.interpolate(ldm_pred_ch0, size=(256,256), mode="bilinear", align_corners=False)
    ldm_mse_pixel = F.mse_loss(ldm_pred_up, raw_target).item()
    ldm_mae_pixel = F.l1_loss(ldm_pred_up, raw_target).item()
    print(f"[LDM ch0 up×4] MSE={ldm_mse_pixel:.6f}, MAE={ldm_mae_pixel:.6f} (ch0 nearest → pixel space, apples-to-oranges; real comparison needs VAE decode)")
    del ldm_wrapped; gc.collect(); torch.cuda.empty_cache()

    # ── 4. CorrDiff ──
    corr_cond = torch.cat([raw_cond, vanilla_pred.detach()], dim=1)  # [1, 3, 256, 256]
    residual_gt = raw_target - vanilla_pred.detach()
    print(f"\n[CorrDiff] residual stats: mean={residual_gt.mean():.4f}, std={residual_gt.std():.4f}, max={residual_gt.abs().max():.4f}")
    corr_wrapped, corr_losses = train_edm(corr_cond, residual_gt)
    corr_residual = edm_sample(corr_wrapped, corr_cond, num_steps=20)
    corr_pred = vanilla_pred.detach() + corr_residual
    corr_mse = F.mse_loss(corr_pred, raw_target).item()
    corr_mae = F.l1_loss(corr_pred, raw_target).item()
    corr_residual_mse = F.mse_loss(corr_residual, residual_gt).item()
    print(f"[CorrDiff] residual MSE={corr_residual_mse:.6f}, final MSE={corr_mse:.6f}, MAE={corr_mae:.6f}")
    del corr_wrapped; gc.collect(); torch.cuda.empty_cache()

    # ── Visualization ──
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(5, 5, figsize=(22, 18))
    vmin, vmax = float(raw_target.min()), float(raw_target.max())

    def imshow(ax, img, title, cmap="Spectral_r"):
        ax.imshow(img.squeeze().cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    def errshow(ax, err, title):
        err_np = err.squeeze().cpu().numpy()
        v = max(abs(err_np.min()), abs(err_np.max()), 1e-6)
        ax.imshow(err_np, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # Row 0: inputs & target
    imshow(axes[0,0], raw_cond[0,0:1], "Input t-1")
    imshow(axes[0,1], raw_cond[0,1:2], "Input t")
    axes[0,2].axis("off")
    imshow(axes[0,3], raw_target[0], "Target (t+1)")
    axes[0,4].axis("off")

    # Row 1: Vanilla UNet
    imshow(axes[1,0], vanilla_pred, "Vanilla UNet pred")
    errshow(axes[1,1], vanilla_pred - raw_target, f"Error (MSE={vanilla_mse:.2e})")
    axes[1,2].axis("off")
    axes[1,3].axis("off")
    axes[1,4].axis("off")

    # Row 2: EDM
    imshow(axes[2,0], edm_pred, "EDM pred (20 steps)")
    errshow(axes[2,1], edm_pred - raw_target, f"Error (MSE={edm_mse:.2e})")
    axes[2,2].axis("off")
    axes[2,3].axis("off")
    axes[2,4].axis("off")

    # Row 3: LDM (latent space)
    imshow(axes[3,0], ldm_pred_ch0, "LDM pred (latent ch0, 20 steps)", cmap="viridis")
    errshow(axes[3,1], ldm_pred_ch0 - lat_target_ch0, f"LDM lat MSE={ldm_mse:.2e} (4ch)")
    axes[3,2].axis("off")
    imshow(axes[3,3], lat_target_ch0, "Latent target (ch0)", cmap="viridis")
    ldm_err_up = F.interpolate((ldm_pred_ch0 - lat_target_ch0), size=(256,256), mode="bilinear", align_corners=False)
    errshow(axes[3,4], ldm_err_up, "Latent error × 16 upsampled")

    # Row 4: CorrDiff
    imshow(axes[4,0], corr_pred, "CorrDiff pred (UNet+residual)")
    errshow(axes[4,1], corr_pred - raw_target, f"Error (MSE={corr_mse:.2e})")
    axes[4,2].axis("off")
    imshow(axes[4,3], residual_gt, "True residual (target-UNet)")
    imshow(axes[4,4], corr_residual, "CorrDiff predicted residual")

    for ax in [axes[0,2], axes[0,4], axes[1,2], axes[1,3], axes[1,4],
               axes[2,2], axes[2,3], axes[2,4], axes[3,2], axes[3,3], axes[3,4]]:
        ax.axis("off")

    fig.suptitle(
        f"Single Sample Overfit Comparison (sample #{SAMPLE_IDX}, {EPOCHS} epochs, lr={LR})",
        fontsize=14, y=0.98
    )
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_ROOT, "overfit_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Loss curves
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for name, losses in [("Vanilla UNet", vanilla_losses), ("EDM", edm_losses),
                         ("LDM (latent)", ldm_losses), ("CorrDiff", corr_losses)]:
        ax2.semilogy(losses, label=name)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss (log scale)")
    ax2.set_title("Training Loss Curves on Single Sample")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(OUTPUT_ROOT, "loss_curves.png")
    fig2.savefig(loss_path, dpi=150); plt.close(fig2)
    print(f"Saved: {loss_path}")

    # Summary table
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"{'Method':<16} {'MSE':<16} {'MAE':<16} {'Notes'}")
    print("-"*70)
    print(f"{'Vanilla UNet':<16} {vanilla_mse:<16.6e} {vanilla_mae:<16.6e}  direct regression")
    print(f"{'EDM':<16} {edm_mse:<16.6e} {edm_mae:<16.6e}  20-step Heun sampling")
    print(f"{'LDM (latent)':<16} {ldm_mse:<16.6e} {ldm_mae:<16.6e}  latent space, 20-step")
    print(f"{'LDM (up×4)':<16} {ldm_mse_pixel:<16.6e} {ldm_mae_pixel:<16.6e}  nearest up → pixel space")
    print(f"{'CorrDiff':<16} {corr_mse:<16.6e} {corr_mae:<16.6e}  UNet prior + residual")
    print("="*70)


if __name__ == "__main__":
    main()