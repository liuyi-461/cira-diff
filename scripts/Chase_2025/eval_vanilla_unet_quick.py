import zarr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os, sys

class ZarrDatasetVal(Dataset):
    def __init__(self, zarr_store):
        self.store = zarr_store
        self.data = zarr.open(self.store, mode='r')
        self.input_images = self.data['input_images']
        self.output_images = self.data['output_images']
        self.num_samples = self.input_images.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        inputs = torch.from_numpy(self.input_images[idx].astype(np.float32))
        target = torch.from_numpy(self.output_images[idx, 0:1].astype(np.float32))
        return inputs, target

CKPT = "/home/group1/26fall_aiclass/ly/cira-diff/outputs/vanilla_unet_full"
VAL_ZARR = "/data1/satcast/edm_GOES_ch13_validation_dataset.zarr"
OUTDIR = "/home/group1/26fall_aiclass/ly/cira-diff/outputs/eval_quick"
os.makedirs(OUTDIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = UNet2DModel.from_pretrained(CKPT)
model.to(device)
model.eval()

val_ds = ZarrDatasetVal(VAL_ZARR)
print(f"Val samples: {len(val_ds)} (single-step target from 18-frame validation zarr)")

mse_all, mae_all = [], []
visualize_indices = [0, 200, 500, 1000]

fig, axes = plt.subplots(len(visualize_indices), 4, figsize=(16, 4 * len(visualize_indices)))

with torch.no_grad():
    for idx in range(len(val_ds)):
        inputs, target = val_ds[idx]
        input_batch = inputs.unsqueeze(0).to(device)
        pred = model(input_batch, torch.zeros(1, device=device)).sample
        pred_np = pred.squeeze(0).cpu().numpy()
        target_np = target.numpy()
        inputs_np = inputs.numpy()

        mse = np.mean((pred_np - target_np) ** 2)
        mae = np.mean(np.abs(pred_np - target_np))
        mse_all.append(mse)
        mae_all.append(mae)

        if idx in visualize_indices:
            row = visualize_indices.index(idx)
            vmin = min(target_np.min(), pred_np.min(), inputs_np.min())
            vmax = max(target_np.max(), pred_np.max(), inputs_np.max())
            cmapper = matplotlib.colormaps.get_cmap('magma')

            ax = axes[row, 0]
            ax.imshow(inputs_np[0], cmap=cmapper, vmin=vmin, vmax=vmax)
            ax.set_title(f't-1')

            ax = axes[row, 1]
            ax.imshow(inputs_np[1], cmap=cmapper, vmin=vmin, vmax=vmax)
            ax.set_title(f't')

            ax = axes[row, 2]
            ax.imshow(pred_np.squeeze(), cmap=cmapper, vmin=vmin, vmax=vmax)
            ax.set_title(f'Pred t+1\nMSE={mse:.4f}')

            ax = axes[row, 3]
            ax.imshow(target_np.squeeze(), cmap=cmapper, vmin=vmin, vmax=vmax)
            ax.set_title(f'True t+1')

            for a in axes[row]:
                a.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "validation_visualization.png"), dpi=120, bbox_inches='tight')
print(f"\nSaved: {OUTDIR}/validation_visualization.png")

mse_all = np.array(mse_all)
mae_all = np.array(mae_all)
print("\n" + "=" * 55)
print(f"Validation set results (n={len(mse_all)}) — single-step teacher-forced")
print("=" * 55)
print(f"MSE  mean={mse_all.mean():.6f}  std={mse_all.std():.6f}  med={np.median(mse_all):.6f}")
print(f"MAE  mean={mae_all.mean():.6f}  std={mae_all.std():.6f}  med={np.median(mae_all):.6f}")

best_idx = np.argsort(mse_all)[:5]
worst_idx = np.argsort(mse_all)[-5:]
print(f"\nBest 5 samples: idx={best_idx}  MSE={mse_all[best_idx]}")
print(f"Worst 5 samples: idx={worst_idx}  MSE={mse_all[worst_idx]}")

np.savez(os.path.join(OUTDIR, "metrics.npz"), mse=mse_all, mae=mae_all,
         best_idx=best_idx, worst_idx=worst_idx)
print(f"\nMetrics saved: {OUTDIR}/metrics.npz")
