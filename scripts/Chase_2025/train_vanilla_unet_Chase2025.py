import torch
import zarr
from torch.utils.data import Dataset
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
import gc
import os 
import torch.distributed as dist
from accelerate import Accelerator
# from huggingface_hub import HfFolder, Repository, whoami #maybe i can delete this 
from tqdm.auto import tqdm
from pathlib import Path
import os
import json
import subprocess
import math
from typing import List, Optional, Tuple, Union
from diffusers.utils.torch_utils import randn_tensor
from torch.utils.tensorboard import SummaryWriter

class ZarrDataset(Dataset):
    """
    This is a new zarr instance of the dataset that loads all data into CPU memory to minimize I/O overhead.
    """
    def __init__(self, zarr_store):
        self.store = zarr_store
        self.data = zarr.open(self.store, mode='r')
        
        # Load data into CPU memory
        self.input_images = torch.tensor(self.data['input_images'][:], dtype=torch.float16, device='cpu')
        self.output_images = torch.tensor(self.data['output_images'][:], dtype=torch.float16, device='cpu')
        self.length = self.input_images.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Access data from memory (still on the CPU)
        return self.output_images[idx], self.input_images[idx]

# Initialize the dataset
zarr_store = '/data1/satcast/edm_GOES_ch13_train_dataset.zarr'
dataset = ZarrDataset(zarr_store)
val_zarr_store = '/data1/satcast/edm_GOES_ch13_validation_dataset.zarr'
val_dataset_heldout = ZarrDataset(val_zarr_store)

import random as _random
_SEED = int(os.environ.get("SEED", "42"))
torch.manual_seed(_SEED)
torch.cuda.manual_seed_all(_SEED)
_random.seed(_SEED)
np.random.seed(_SEED)
splits = torch.utils.data.random_split(dataset, [0.8, 0.2])

ds_train = splits[0]
ds_val = splits[1]

_BS = 24

train_dataloader = torch.utils.data.DataLoader(
    ds_train, batch_size=_BS, shuffle=True, num_workers=4, pin_memory=True
)

val_dataloader = torch.utils.data.DataLoader(
    ds_val, batch_size=_BS, shuffle=False, num_workers=4, pin_memory=True
)

val_heldout_dataloader = torch.utils.data.DataLoader(
    val_dataset_heldout, batch_size=_BS, shuffle=False, num_workers=4, pin_memory=True
)

print(f"Train samples: {len(ds_train)}, held-out val (80/20 split): {len(ds_val)}, independent val zarr: {len(val_dataset_heldout)}")

# ################### \Imports ########################


# ################### Classes ########################

@dataclass
class TrainingConfig:
    """ This should be probably in some sort of config file, but for now its here... """
    image_size = 256  
    train_batch_size = 24
    val_batch_size = 24
    num_epochs = 210
    gradient_accumulation_steps = 2
    learning_rate = 1e-4 
    lr_warmup_steps = 500
    save_model_epochs = 1 
    mixed_precision = "fp16" 
    output_dir = os.environ.get("OUTPUT_DIR", "/home/group1/26fall_aiclass/ly/cira-diff/outputs/vanilla_unet_full/")
    push_to_hub = False
    hub_private_repo = False
    overwrite_output_dir = True  
    seed = 0 

# ################### \Classes ########################

# ################### Funcs ########################

def train_loop(config, model, optimizer, train_dataloader, lr_scheduler, val_dataloader, val_heldout_dataloader=None):
    """ 
    This is the main show! the training loop 
    """
    
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    accelerator.wait_for_everyone()

    state_file = os.path.join(config.output_dir, "training_state.json")
    accelerator_state_exists = os.path.exists(os.path.join(config.output_dir, "optimizer.bin"))
    resume = accelerator_state_exists and os.path.exists(state_file)

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

        writer = SummaryWriter(os.path.join(config.output_dir, "logs/images"))

        for step, batch in enumerate(train_dataloader):
            clean_images_eval = batch[0].to(accelerator.device)
            condition_images_eval = batch[1].to(accelerator.device)
            break 
                    
        image = condition_images_eval[0,0:1].squeeze(0).unsqueeze(-1).cpu()
        color_image = torch.tensor(colorize(image,vmin=-4,vmax=2,cmap='Spectral_r')).permute(2, 0, 1)
        writer.add_image("Training Data", color_image, 0)
        
        image = condition_images_eval[0,1:2].squeeze(0).unsqueeze(-1).cpu()
        color_image = torch.tensor(colorize(image,vmin=-4,vmax=2,cmap='Spectral_r')).permute(2, 0, 1)
        writer.add_image("Training Data", color_image, 1)

        image = clean_images_eval[0,0:1].squeeze(0).unsqueeze(-1).cpu()
        color_image = torch.tensor(colorize(image,vmin=-4,vmax=2,cmap='Spectral_r')).permute(2, 0, 1)
        writer.add_image("Training Data", color_image, 2)
        writer.add_image("Slider", color_image, 0)
        
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    accelerator.wait_for_everyone()

    if resume:
        accelerator.load_state(config.output_dir)
        with open(state_file) as f:
            state = json.load(f)
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_moving_val_loss = state["best_moving_val_loss"]
        best_internal_val_loss = state.get("best_internal_val_loss", float('inf'))
        best_internal_epoch = state.get("best_internal_epoch", -1)
        no_improvement_count = state["no_improvement_count"]
        loss_history = state["loss_history"]
        if accelerator.is_main_process:
            print(f"[Resume] epoch {state['epoch']} -> start_epoch {start_epoch}, global_step={global_step}, best_internal_val={best_internal_val_loss:.6f}")
    else:
        start_epoch = 0
        global_step = 0
        best_moving_val_loss = float('inf')
        best_internal_val_loss = float('inf')
        best_internal_epoch = -1
        no_improvement_count = 0
        loss_history = []

    if accelerator.is_main_process and not resume:
        try:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL).decode().strip()
            git_branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo_root, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            git_commit = "unknown"
            git_branch = "unknown"
        run_meta = {
            "git_commit": git_commit,
            "git_branch": git_branch,
            "dataset_sizes": {"train": len(ds_train), "internal_val": len(ds_val), "independent_val": len(val_dataset_heldout)},
            "split_seed": 42,
            "normalization": {"mean": 0.0, "std": 1.0},
            "architecture": "UNet2DModel",
            "in_channels": 2, "out_channels": 1,
            "layers_per_block": 2,
            "block_out_channels": (128, 128, 256, 256, 512, 512),
            "down_block_types": ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            "optimizer": "AdamW", "learning_rate": config.learning_rate,
            "lr_scheduler": "cosine_schedule_with_warmup",
            "lr_warmup_steps": config.lr_warmup_steps,
            "batch_size": config.train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.train_batch_size * config.gradient_accumulation_steps,
            "num_epochs": config.num_epochs,
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 1e-6,
            "early_stopping_window_size": 5,
            "early_stopping_criterion": "internal_val_moving_average",
            "mixed_precision": config.mixed_precision,
            "loss": "MSE",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
            "resume": False,
        }
        with open(os.path.join(config.output_dir, "run_metadata.json"), "w") as f:
            json.dump(run_meta, f, indent=2)
        print(f"[Run metadata] written to {os.path.join(config.output_dir, 'run_metadata.json')}")

    patience = 10
    min_delta = 1e-6
    window_size = 5
    loss_fn = torch.nn.MSELoss(reduction='none')
    
     
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        epoch_loss = torch.tensor(0.0, device=accelerator.device)
        
        for step, batch in enumerate(train_dataloader):
            clean_images = batch[0]
            condition_images = batch[1]
            
            with accelerator.accumulate(model):
                yhat = model(condition_images, torch.zeros(condition_images.shape[0]).to(accelerator.device), return_dict=False)[0]
                loss = loss_fn(clean_images.to(torch.float), yhat.to(torch.float)).mean()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.detach()
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        epoch_loss = accelerator.gather(epoch_loss)
        mean_epoch_loss = epoch_loss.sum() / (len(train_dataloader) * accelerator.num_processes)
        
        model.eval()
        
        # --- Internal val (80/20 split, early stopping criterion) ---
        internal_val_loss_acc = torch.tensor(0.0, device=accelerator.device)
        with torch.no_grad():
            for batch in tqdm(val_dataloader, total=len(val_dataloader), desc=f"Internal Val", disable=not accelerator.is_local_main_process):
                clean_images = batch[0].to(accelerator.device)
                condition_images = batch[1].to(accelerator.device)
                yhat = model(condition_images, torch.zeros(condition_images.shape[0], device=accelerator.device), return_dict=False)[0]
                loss = loss_fn(clean_images.to(torch.float), yhat.to(torch.float)).mean()
                internal_val_loss_acc += loss.detach()
        mean_internal_val_loss = internal_val_loss_acc / len(val_dataloader)

        # --- Independent val (validation zarr 1024 samples, diagnostic only) ---
        mean_independent_val_loss = float('nan')
        if val_heldout_dataloader is not None:
            independent_val_loss_acc = torch.tensor(0.0, device=accelerator.device)
            with torch.no_grad():
                for batch in tqdm(val_heldout_dataloader, total=len(val_heldout_dataloader), desc=f"Independent Val", disable=not accelerator.is_local_main_process):
                    clean_images = batch[0][:, 0:1].to(accelerator.device)
                    condition_images = batch[1].to(accelerator.device)
                    yhat = model(condition_images, torch.zeros(condition_images.shape[0], device=accelerator.device), return_dict=False)[0]
                    loss = loss_fn(clean_images.to(torch.float), yhat.to(torch.float)).mean()
                    independent_val_loss_acc += loss.detach()
            mean_independent_val_loss = (independent_val_loss_acc / len(val_heldout_dataloader)).item()
        
        # Log to TB
        logs = {
            "epoch_loss": mean_epoch_loss.item(),
            "epoch": epoch,
            "internal_val_loss": mean_internal_val_loss.item(),
            "independent_val_loss": mean_independent_val_loss,
        }
        accelerator.log(logs, step=epoch)

        if accelerator.is_main_process:
            print(f"[Epoch {epoch:3d}] train={mean_epoch_loss.item():.6f}  "
                  f"internal_val={mean_internal_val_loss.item():.6f}  "
                  f"independent_val={mean_independent_val_loss:.6f}  "
                  f"global_step={global_step}")

        # Early stopping: based on internal val moving average (论文 protocol, 严格不变)
        loss_history.append(mean_internal_val_loss.item())
        if len(loss_history) >= window_size:
            moving_average = sum(loss_history[-window_size:]) / window_size
            logs = {"moving_internal_val_loss": moving_average, "epoch": epoch}
            accelerator.log(logs, step=epoch)
            if accelerator.is_main_process:
                print(f"  moving_internal_val (win={window_size})={moving_average:.6f}  "
                      f"best_moving={best_moving_val_loss:.6f}  "
                      f"no_improve={no_improvement_count}/{patience}")

            if moving_average < (best_moving_val_loss - min_delta):
                best_moving_val_loss = moving_average
                no_improvement_count = 0
            else:
                no_improvement_count += 1

        # --- Save everything ---
        if accelerator.is_main_process:
            accelerator.save_state(config.output_dir)
            accelerator.unwrap_model(model).save_pretrained(config.output_dir)

            if mean_internal_val_loss.item() < best_internal_val_loss - 1e-12:
                best_internal_val_loss = mean_internal_val_loss.item()
                best_internal_epoch = epoch
                best_dir = os.path.join(config.output_dir, "best_internal_unet")
                os.makedirs(best_dir, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(best_dir)
                print(f"  [Best checkpoint] internal_val improved to {best_internal_val_loss:.6f} (epoch {epoch}) -> {best_dir}")

            state = {
                "epoch": epoch,
                "global_step": global_step,
                "best_moving_val_loss": best_moving_val_loss,
                "best_internal_val_loss": best_internal_val_loss,
                "best_internal_epoch": best_internal_epoch,
                "no_improvement_count": no_improvement_count,
                "loss_history": loss_history,
            }
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

            # TensorBoard image visualization
            with torch.no_grad():
                images_batch = model(condition_images, torch.zeros(condition_images.shape[0], device=accelerator.device), return_dict=False)[0]
            image = images_batch[0].squeeze(0).unsqueeze(-1).cpu().numpy()
            color_image = torch.tensor(colorize(image, vmin=-4, vmax=2, cmap='Spectral_r')).permute(2, 0, 1)
            writer.add_image("Output Image", color_image, epoch)
            writer.add_image("Slider", color_image, 1)
            image = clean_images_eval[0, 0:1].squeeze(0).unsqueeze(-1).cpu()
            color_image = torch.tensor(colorize(image, vmin=-4, vmax=2, cmap='Spectral_r')).permute(2, 0, 1)
            writer.add_image("Slider", color_image, 0)

        accelerator.wait_for_everyone()

        if no_improvement_count >= patience:
            if accelerator.is_main_process:
                print(f"\n[Early stopping] triggered after {patience} epochs without improvement. "
                      f"best_internal_val={best_internal_val_loss:.6f} at epoch {best_internal_epoch}")
            accelerator.wait_for_everyone()
            break
                    
        gc.collect()
        
import matplotlib
import matplotlib.cm

def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for Torch/Numpy that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: Matplotlib default colormap)
    
    Returns a 4D uint8 tensor of shape [height, width, 4].
    """

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    value = value.squeeze()

    cmapper = matplotlib.colormaps.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)
    return value

# ################### \Funcs ########################

# ################### CODE ########################

#initalize config 
config = TrainingConfig()

# go ahead and build a UNET, this was the exact same as the butterfly example, but different channels. This is a big model.. 
model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=2,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

#left this the same as the butterfly example 
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)


#main method here! 
train_loop(config, model, optimizer, train_dataloader, lr_scheduler, val_dataloader, val_heldout_dataloader)