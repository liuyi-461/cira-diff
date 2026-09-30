#################### Imports ########################

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
import json
import os 
import subprocess
import torch.distributed as dist
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
import os
import math
from typing import List, Optional, Tuple, Union
from diffusers.utils.torch_utils import randn_tensor
from torch.utils.tensorboard import SummaryWriter


#################### \Imports ########################


#################### Classes ########################

@dataclass
class TrainingConfig:
    image_size = 256
    train_batch_size = 22
    val_batch_size = 22
    num_epochs = 1000
    gradient_accumulation_steps = 4
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_model_epochs = 1
    mixed_precision = "fp16"
    output_dir = "/home/group1/26fall_aiclass/ly/cira-diff/outputs/corrdiff_full/"
    push_to_hub = False
    hub_private_repo = False
    overwrite_output_dir = True
    seed = 0
    dataset_path = "/data1/satcast/edm_GOES_ch13_train_dataset_CorrDiff.zarr"
    val_heldout_path = "/data1/satcast/edm_GOES_ch13_val_dataset_CorrDiff.zarr"
    plot_images = True
    images_idx = [3, 5, 10, 15]
    P_mean = -1.2
    P_std = 1.2
    sigma_data = 0.5
    
    
    


class EDMPrecond(torch.nn.Module):
    """ Original Func:: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/networks.py#L519
    
    This is a wrapper for your diffusers model. It's purpose is to apply the preconditioning that is talked about in Karras et al. (2022)'s EDM paper. 
    
    I've made some changes for the sake of conditional-EDM (the original paper is unconditional).
    
    """
    def __init__(self,
        generation_channels,                # number of channels you want to generate
        model,                              # pytorch model from diffusers 
        use_fp16        = True,             # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data. this was the default from above
    ):
        super().__init__()
        self.generation_channels = generation_channels
        self.model = model
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        
    def forward(self, x, sigma, force_fp32=False, **model_kwargs):
        
        """ 
        
        This method is to 'call' the neural net. But this is the preconditioning from the Karras EDM paper. 
        
        note for conditional, it expects x to have the condition in the channel dim (dim=1). and the images you want to generate should already have noise.
        
        x: input stacked image with the generation images stacked with the condition images [batch,generation_channels + condition_channels,nx,ny]
        sigma: the noise level of the images in batch [??]
        force_fp32: this is forcing calculations to be a certain percision. 
        
        """
        
        #for the calculations, use float 32
        x = x.to(torch.float32)
        #reshape sigma from _ to _ 
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        
        #forcing dtype matching
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        
        #get weights from EDM 
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        # split out the images you want to generate and the condition, because the scaling will depend on this. 
        x_noisy = torch.clone(x[:, 0:self.generation_channels])
        
        #the condition
        x_condition = torch.clone(x[:, self.generation_channels:])

        
        #concatinate back with the scaling applied to only the the generation dimension (x_noisy)
        model_input_images = torch.cat([x_noisy*c_in, x_condition], dim=1)
        
        #denoise the image (e.g., run it through your diffusers model) 
        F_x = self.model((model_input_images).to(dtype), c_noise.flatten(), return_dict=False)[0]
        
        #force dtype
        assert F_x.dtype == dtype
        
        #apply additional scalings: make sure you apply skip just to the generation dim (x[:,0:generation_channel]) and NOT applied to (x*c_in)
        D_x = c_skip * x_noisy + c_out * F_x.to(torch.float32)
        
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class EDMLoss:
    
    """Original Func:: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/loss.py
    
    This is the loss function class from Karras et al. (2022)'s EDM paper. Only thing changed here is that the __call__ takes the clean_images and the condition_images seperately. It expects your model to be wrapped with that EDMPrecond class. 
    
    """
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        """ These describe the distribution of sigmas we should sample during training """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, clean_images, condition_images, labels=None, augment_pipe=None):
        
        """ 
        
        net: is a pytorch model wrapped with EDMPrecond
        clean_images: the images you want to generate, [batch,generation_channels,nx,ny]
        condition_images:images you want to condition with [batch,condition_channels,nx,ny]
        
        """
        
        #get random seeds, one for each image in the batch 
        rnd_normal = torch.randn([clean_images.shape[0], 1, 1, 1], device=clean_images.device)
        
        #get random noise levels (sigmas)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        
        #get the loss weight for those sigmas 
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        #make the noise scalars images so we can add them to our images
        n = torch.randn_like(clean_images) * sigma
    
        #add noise to the clean images 
        noisy_images = torch.clone(clean_images + n)
        
        #cat the images for the wrapped model call 
        model_input_images = torch.cat([noisy_images, condition_images], dim=1)
        
        #call the EDMPrecond model 
        denoised_images = net(model_input_images, sigma)
        
        #calc the weighted loss at each pixel, the mean across all GPUs and pixels is in the main train_loop 
        loss = weight * ((denoised_images - clean_images) ** 2)
        
        return loss
    

class ZarrDataset(Dataset):
    def __init__(self, zarr_store):
        self.store = zarr_store
        self.data = zarr.open(self.store, mode='r')
        self.length = self.data['input_images'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_image = self.data['input_images'][idx]
        output_image = self.data['output_images'][idx]
        return torch.tensor(output_image, dtype=torch.float32), torch.tensor(input_image, dtype=torch.float32)


class ZarrDatasetVal(Dataset):
    def __init__(self, zarr_store):
        self.store = zarr_store
        self.data = zarr.open(self.store, mode='r')
        self.length = self.data['input_images'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_image = self.data['input_images'][idx]
        output_image = self.data['output_images'][idx, 0]
        return torch.tensor(output_image, dtype=torch.float32).unsqueeze(0), torch.tensor(input_image, dtype=torch.float32)

#################### \Classes ########################

#################### Funcs ########################

def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count())) 
    
def train_loop(config, model, optimizer, train_dataloader, lr_scheduler, val_heldout_dataloader=None, run_metadata=None):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    state_file = os.path.join(config.output_dir, "training_state.json")
    accelerator_state_exists = os.path.exists(os.path.join(config.output_dir, "optimizer.bin"))
    resume = accelerator_state_exists and os.path.exists(state_file)

    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("corrdiff_train")
        if run_metadata is not None:
            run_metadata["resume"] = resume
            with open(os.path.join(config.output_dir, "run_metadata.json"), "w") as f:
                json.dump(run_metadata, f, indent=2, default=str)
            print(f"[Run metadata] written to {os.path.join(config.output_dir, 'run_metadata.json')}")

    data_mean = torch.tensor(-0.0009)
    data_std = torch.tensor(0.0807)

    if config.plot_images and val_heldout_dataloader is not None:
        writer = SummaryWriter(os.path.join(config.output_dir, "logs/images"))
        rnd = StackedRandomGenerator(accelerator.device, np.arange(0, config.train_batch_size, 1).astype(int).tolist())
        latents = rnd.randn([config.train_batch_size, 1, 256, 256], device=accelerator.device)
        plot_iter = iter(val_heldout_dataloader)
        clean_images_eval, condition_images_eval = next(plot_iter)
        clean_images_eval = clean_images_eval.to(accelerator.device)
        condition_images_eval = condition_images_eval.to(accelerator.device)
        residual_truth_eval = clean_images_eval * data_std + data_mean
        for i in np.arange(0, len(config.images_idx)):
            image = condition_images_eval[config.images_idx[i], 0:1].squeeze(0).unsqueeze(-1).cpu()
            color_image = torch.tensor(colorize(image, vmin=-6, vmax=4, cmap='Spectral_r')).permute(2, 0, 1)
            writer.add_image("UNet forecast (cond ch0)", color_image, 0)
            image = condition_images_eval[config.images_idx[i], 1:2].squeeze(0).unsqueeze(-1).cpu()
            color_image = torch.tensor(colorize(image, vmin=-6, vmax=4, cmap='Spectral_r')).permute(2, 0, 1)
            writer.add_image("CorrDiff forecast (cond ch1)", color_image, 1)
            image = condition_images_eval[config.images_idx[i], 2:3].squeeze(0).unsqueeze(-1).cpu()
            color_image = torch.tensor(colorize(image, vmin=-6, vmax=4, cmap='Spectral_r')).permute(2, 0, 1)
            writer.add_image("t-1 forecast (cond ch2)", color_image, 2)
            image = residual_truth_eval[config.images_idx[i], 0:1].squeeze(0).unsqueeze(-1).cpu()
            color_image = torch.tensor(colorize(image, vmin=-6, vmax=4, cmap='Spectral_r')).permute(2, 0, 1)
            writer.add_image("Residual truth", color_image, 3)
            full_forecast = condition_images_eval[config.images_idx[i], 0:1].squeeze(0).unsqueeze(-1).cpu() + residual_truth_eval[config.images_idx[i], 0:1].squeeze(0).unsqueeze(-1).cpu()
            color_image = torch.tensor(colorize(full_forecast, vmin=-6, vmax=4, cmap='Spectral_r')).permute(2, 0, 1)
            writer.add_image("Full forecast (UNet + residual truth)", color_image, 4)

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    best_independent_val_loss = float('inf')
    best_independent_epoch = -1

    if resume:
        accelerator.load_state(config.output_dir)
        with open(state_file) as f:
            state = json.load(f)
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_independent_val_loss = state.get("best_independent_val_loss", float('inf'))
        best_independent_epoch = state.get("best_independent_epoch", -1)
        print(f"[Resume] loaded from {config.output_dir}, resuming at epoch {start_epoch}, global_step={global_step}, best_independent_val={best_independent_val_loss:.6f} at epoch {best_independent_epoch}")
    else:
        start_epoch = 0
        global_step = 0
        print(f"[Fresh start] training from epoch 0, no prior checkpoint")

    loss_fn = EDMLoss(P_mean=config.P_mean, P_std=config.P_std, sigma_data=config.sigma_data)

    for epoch in range(start_epoch, config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        epoch_loss = torch.tensor(0.0, device=accelerator.device)

        for step, batch in enumerate(train_dataloader):
            clean_images = batch[0]
            condition_images = batch[1]
            with accelerator.accumulate(model):
                per_sample_loss = loss_fn(model, clean_images, condition_images)
                loss = per_sample_loss.mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
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
        accelerator.log({"epoch_loss": mean_epoch_loss.item(), "epoch": epoch}, step=epoch)

        mean_independent_val_loss = float('nan')
        if val_heldout_dataloader is not None:
            unwrapped_model = accelerator.unwrap_model(model)
            independent_val_loss_acc = torch.tensor(0.0, device=accelerator.device)
            n_batches = 0
            with torch.no_grad():
                for batch in tqdm(val_heldout_dataloader, total=len(val_heldout_dataloader),
                                  desc=f"Independent Val", disable=not accelerator.is_local_main_process):
                    clean_v = batch[0].to(accelerator.device)
                    cond_v = batch[1].to(accelerator.device)
                    sigma_min_val = 0.002
                    sigma_val = torch.ones(cond_v.shape[0], 1, 1, 1, device=accelerator.device) * sigma_min_val
                    model_input = torch.cat([clean_v, cond_v], dim=1)
                    pred = unwrapped_model(model_input, sigma_val)
                    independent_val_loss_acc += F.mse_loss(pred, clean_v).detach()
                    n_batches += 1
            if n_batches > 0:
                mean_independent_val_loss = (independent_val_loss_acc / n_batches).item()
                accelerator.log({"independent_val_loss": mean_independent_val_loss, "epoch": epoch}, step=epoch)

        if accelerator.is_main_process:
            if not (mean_independent_val_loss != mean_independent_val_loss):
                if mean_independent_val_loss < best_independent_val_loss - 1e-12:
                    best_independent_val_loss = mean_independent_val_loss
                    best_independent_epoch = epoch
                    best_dir = os.path.join(config.output_dir, "best_corrdiff")
                    os.makedirs(best_dir, exist_ok=True)
                    accelerator.unwrap_model(model).model.save_pretrained(best_dir)
                    print(f"  [Best checkpoint] independent_val improved to {best_independent_val_loss:.6f} (epoch {epoch}) -> {best_dir}")

            state = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": mean_epoch_loss.item(),
                "independent_val_loss": mean_independent_val_loss,
                "best_independent_val_loss": best_independent_val_loss,
                "best_independent_epoch": best_independent_epoch,
            }
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
            accelerator.save_state(config.output_dir)
            accelerator.unwrap_model(model).model.save_pretrained(config.output_dir)
            print(f"[Checkpoint] saved epoch {epoch} to {config.output_dir}  |  best_independent_val={best_independent_val_loss:.6f} at epoch {best_independent_epoch}")

            if config.plot_images:
                images_batch = edm_sampler(model, latents, condition_images_eval, num_steps=18)
                residual_pred = images_batch * data_std + data_mean
                for i in np.arange(0, len(config.images_idx)):
                    image = residual_pred[config.images_idx[i]].squeeze(0).unsqueeze(-1).cpu().numpy()
                    color_image = torch.tensor(colorize(image, vmin=-3, vmax=3, cmap='Spectral_r')).permute(2, 0, 1)
                    writer.add_image("Residual prediction", color_image, epoch)

        gc.collect()

def edm_sampler(net, latents, condition_images, randn_like=torch.randn_like,num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    """ adapted from: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/generate.py 
    
    only thing i had to change was provide a condition as input to this func, then take that input and concat with generated image for the model call. 
    
    net: expects a wrapped diffusers model with the EDMPrecond
    latents: a noise seed with the same shape as condition_images
    condition_images: the condition, [batch or ens_size,condition_channels,nx,ny]
    randn_like: how to generate randomness
    num_steps: the number of generation steps you want to take (note model calls are ~2x this because the second order correction)
    sigma_min: smallest amount of noise 
    sigma_max: largest amount of noise 
    rho: related to the step size with time ??? 
    S_churn: how much stocasisty you want to add to the process 
    S_min: min sigma step of when to add the stocastic bit 
    S_max: max sigma step of when to add the stocastic bit  
    S_noise: scale to the noise we add in the stocastic bit 
    
    """
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        #need to concat the condition here 
        model_input_images = torch.cat([x_hat, condition_images], dim=1)
        # Euler step.
        with torch.no_grad():
            denoised = net(model_input_images, t_hat).to(torch.float64)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            model_input_images = torch.cat([x_next, condition_images], dim=1)
            with torch.no_grad():
                denoised = net(model_input_images, t_next).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

class StackedRandomGenerator:  # pragma: no cover
    """
    adapted from: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/generate.py 
    
    Wrapper for torch.Generator that allows specifying a different random seed
    for each sample in a minibatch.
    """

    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError(
                f"Expected first dimension of size {len(self.generators)}, got {size[0]}"
            )
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError(
                f"Expected first dimension of size {len(self.generators)}, got {size[0]}"
            )
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )

import matplotlib
import matplotlib.cm

def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    from here: https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b 
    
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

def save_checkpoint(model, optimizer, lr_scheduler, epoch, step, accelerator, checkpoint_path):
    """ A function from chatGPT to help checkpoint out things for training restarts """
    checkpoint = {
        'model_state_dict': accelerator.unwrap_model(model).model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
    }
    if accelerator.is_main_process:
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}, step {step}.")
    
def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, accelerator):
    """ A function from chatGPT to help load checkpoints training restarts """ 
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)
    accelerator.unwrap_model(model).model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    print(f"Checkpoint loaded. Resuming from epoch {epoch}, step {step}.")
    return epoch, step

#################### \Funcs ########################

import subprocess

config = TrainingConfig()

dataset = ZarrDataset(config.dataset_path)
val_dataset_heldout = ZarrDatasetVal(config.val_heldout_path)

print(f"Train samples: {len(dataset)}  independent val zarr: {len(val_dataset_heldout)}")

train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_heldout_dataloader = torch.utils.data.DataLoader(
    val_dataset_heldout, batch_size=config.val_batch_size, shuffle=False, num_workers=4, pin_memory=True
)

model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=4,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

model_wrapped = EDMPrecond(1, model)

optimizer = torch.optim.AdamW(model_wrapped.model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

git_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
run_metadata = {
    "git_commit": git_commit,
    "git_branch": git_branch,
    "dataset_sizes": {
        "train": len(dataset),
        "independent_val": len(val_dataset_heldout),
    },
    "normalization": {"mean": -0.0009, "std": 0.0807},
    "architecture": "CorrDiff: EDMPrecond(UNet2DModel) predicting residual (forecast - truth)",
    "in_channels": 4,
    "condition_channels": 3,
    "condition_semantics": ["UNet forecast", "CorrDiff forecast", "t-1 forecast"],
    "output_channels": 1,
    "output_semantics": "residual (forecast - truth)",
    "layers_per_block": 2,
    "block_out_channels": [128, 128, 256, 256, 512, 512],
    "optimizer": "AdamW",
    "learning_rate": config.learning_rate,
    "lr_scheduler": "cosine_schedule_with_warmup",
    "lr_warmup_steps": config.lr_warmup_steps,
    "batch_size": config.train_batch_size,
    "gradient_accumulation_steps": config.gradient_accumulation_steps,
    "effective_batch_size": config.train_batch_size * config.gradient_accumulation_steps,
    "num_epochs": config.num_epochs,
    "training_protocol": "fixed_epochs_with_independent_val_best_checkpoint",
    "mixed_precision": config.mixed_precision,
    "edm_P_mean": config.P_mean,
    "edm_P_std": config.P_std,
    "edm_sigma_data": config.sigma_data,
    "slurm_job_id": subprocess.check_output(['bash', '-c', 'echo ${SLURM_JOB_ID:-N/A}']).decode().strip(),
}

train_loop(config, model_wrapped, optimizer, train_dataloader, lr_scheduler,
           val_heldout_dataloader=val_heldout_dataloader, run_metadata=run_metadata)