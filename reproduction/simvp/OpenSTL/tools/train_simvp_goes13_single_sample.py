"""Single-sample overfit training + forecast visualization for SimVP (GOES-16 ABI ch13).

Purpose
-------
Complement to ``tools/train_simvp_goes13_smoke.py``. That script validates the
*pipeline* on many samples; this one validates the *model itself*: pick ONE sample
and let SimVP overfit it to convergence. A model whose structure is wired up
correctly must be able to drive a single-sample loss to near zero and reproduce the
target frame; failure modes that do **not** show up in aggregate metrics (channel
mismatch, wrong time axis, broken skip connection, loss on the wrong tensor) show up
here immediately in the side-by-side figure.

Outputs (default dir ``./single_sample_goes13_output``):
    comparison.png   condition t-2 / condition t-1 / truth / prediction / error
    loss_curve.png   per-epoch training loss
    *.npy            raw (already de-normalized) arrays
    summary.json     run configuration + metrics, for the evidence package

Usage
-----
    python tools/train_simvp_goes13_single_sample.py \
        --zarr /path/to/satcast/edm_GOES_ch13_train_dataset.zarr \
        --index 0 --epochs 100 --lr 1e-3

Notes
-----
* Reuses ``openstl/datasets/dataloader_goes13.py`` for the data contract.
* Normalization statistics are taken from a reference pool (``--ref-samples``,
  default 32) rather than from the single sample itself, which would degenerate.
* Visualization styling follows ``test_dl/test_single_sample_Chase2025.py``
  (``Spectral_r``, vmin=-4, vmax=2) so the two views stay comparable.

Dependencies: torch, lightning, timm, fvcore, zarr, numpy, matplotlib.
"""

import json
import os
import os.path as osp
import warnings

warnings.filterwarnings('ignore')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
import lightning as l
from lightning import Trainer

from openstl.api import BaseExperiment
from openstl.datasets.dataloader_goes13 import (
    AFT_SEQ_LENGTH, CHANNELS, DATA_NAME, GOES13Dataset, IMAGE_SIZE, PRE_SEQ_LENGTH,
    sample_indices,
)
from openstl.datasets.utils import create_loader
from openstl.utils import create_parser, default_parser, load_config, update_config

DEFAULT_ZARR = '/data1/satcast/edm_GOES_ch13_train_dataset.zarr'
HERE = osp.dirname(osp.abspath(__file__))
DEFAULT_CONFIG = osp.join(HERE, '..', 'configs', 'goes13', 'simvp', 'SimVP_gSTA.py')


# --------------------------------------------------------------- callbacks ---
class LossHistoryCallback(l.Callback):
    """Collect one training loss value per epoch."""

    def __init__(self, sink=None):
        super().__init__()
        self.sink = sink if sink is not None else []

    def on_train_epoch_end(self, trainer, pl_module):
        value = trainer.callback_metrics.get('train_loss')
        if value is not None:
            self.sink.append(float(value.detach().cpu()))


class SingleSampleExperiment(BaseExperiment):
    """BaseExperiment with documented deviation D1 (see smoke script) + loss logging."""

    def __init__(self, *args, **kwargs):
        self.loss_history = []
        super().__init__(*args, **kwargs)

    def _load_callbacks(self, args, save_dir, ckpt_dir):
        callbacks, save_dir = super()._load_callbacks(args, save_dir, ckpt_dir)
        callbacks.append(LossHistoryCallback(self.loss_history))
        return callbacks, save_dir

    def _init_trainer(self, args, callbacks, strategy):
        # Upstream hardcodes accelerator='gpu'; follow the machine instead so the
        # check also runs on CPU-only boxes.
        use_gpu = torch.cuda.is_available() and str(args.device).startswith('cuda')
        return Trainer(
            devices=args.gpus if use_gpu else 1,
            max_epochs=args.epoch,
            strategy=strategy,
            accelerator='gpu' if use_gpu else 'cpu',
            callbacks=callbacks,
            enable_progress_bar=False,
        )


# ------------------------------------------------------------------- utils ---
def colorize(value, vmin=None, vmax=None, cmap='Spectral_r'):
    """Same helper as test_dl/test_single_sample_Chase2025.py."""
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) if vmin != vmax else value * 0.0
    value = value.squeeze()
    return matplotlib.colormaps[cmap](value, bytes=True)[..., :3]


def save_panel(images, titles, path, cmap='Spectral_r', vmin=-4, vmax=2):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.4))
    axes = [axes] if n == 1 else axes
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(colorize(img, vmin=vmin, vmax=vmax, cmap=cmap))
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def denorm(array, mean_value, std_value):
    """Undo the z-score normalization applied by GOES13Dataset."""
    return np.asarray(array, dtype=np.float32) * std_value + mean_value


def scalar_metrics(pred, true):
    diff = pred - true
    mse = float(np.mean(diff ** 2))
    ss_tot = float(((true - true.mean()) ** 2).sum())
    ss_res = float((diff ** 2).sum())
    return {
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
        'mae': float(np.mean(np.abs(diff))),
        'r2': float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan'),
        'bias': float(diff.mean()),
    }


# -------------------------------------------------------------------- cli ---
def make_cli_parser():
    parser = create_parser()
    parser.description = 'SimVP single-sample overfit + forecast visualization (GOES-13)'
    parser.add_argument('--zarr', default=DEFAULT_ZARR, type=str)
    parser.add_argument('--config', default=DEFAULT_CONFIG, type=str,
                        help='Optional OpenSTL config file for model hyper-parameters')
    parser.add_argument('--index', type=int, default=0, help='Sample index to overfit')
    parser.add_argument('--ref-samples', type=int, default=32,
                        help='Samples used to estimate normalization mean/std')
    parser.add_argument('--epochs', type=int, default=100, help='Overfit epochs')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model-type', default='gSTA', type=str)
    parser.add_argument('--vmin', type=float, default=-4.0, help='Colorize lower bound')
    parser.add_argument('--vmax', type=float, default=2.0, help='Colorize upper bound')
    parser.add_argument('--output-dir', default='./single_sample_goes13_output', type=str)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--display-method-info', action='store_true', default=False)
    return parser


def build_args(cli):
    args = create_parser().parse_args([])

    # 1) model hyper-parameters from the OpenSTL config file (hid_S/hid_T/N_S/N_T/...)
    if cli.config and osp.isfile(cli.config):
        update_config(args.__dict__, load_config(cli.config), exclude_keys=['method'])
    else:
        print(f'warning: config not found at {cli.config}; falling back to model defaults')

    # 2) documented defaults for anything the config did not provide
    for key, value in default_parser().items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if not hasattr(args, 'ckpt_path'):
        args.ckpt_path = None

    overrides = {
        'method': 'SimVP',
        'dataname': 'goes13',
        'data_name': DATA_NAME,
        'model_type': cli.model_type,
        'in_shape': [PRE_SEQ_LENGTH, CHANNELS, IMAGE_SIZE, IMAGE_SIZE],
        'pre_seq_length': PRE_SEQ_LENGTH,
        'aft_seq_length': AFT_SEQ_LENGTH,
        'total_length': PRE_SEQ_LENGTH + AFT_SEQ_LENGTH,
        'metrics': ['mse', 'mae', 'rmse'],
        'epoch': cli.epochs,
        'batch_size': 1,
        'val_batch_size': 1,
        'num_workers': 0,
        'lr': cli.lr,
        'seed': cli.seed,
        'gpus': list(cli.gpus),
        'ex_name': cli.ex_name,
        'res_dir': cli.res_dir,
        'device': 'cpu' if cli.cpu or not torch.cuda.is_available() else 'cuda',
        'no_display_method_info': not cli.display_method_info,
    }
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# ------------------------------------------------------------------- main ---
def main():
    cli = make_cli_parser().parse_args()
    os.makedirs(cli.output_dir, exist_ok=True)

    store = zarr.open(cli.zarr, mode='r')
    n_samples = store['output_images'].shape[0]
    if not 0 <= cli.index < n_samples:
        raise IndexError(f'--index {cli.index} out of range for {n_samples} samples')

    args = build_args(cli)

    print('=' * 78)
    print('SimVP single-sample overfit — EDM GOES-16 ABI ch13 (2 -> 1)')
    print('=' * 78)
    print(f'zarr store   : {cli.zarr}  ({n_samples} samples)')
    print(f'overfit idx  : {cli.index}   epochs={args.epoch}  lr={args.lr}')
    print(f'model        : {args.method} / {args.model_type}  in_shape={args.in_shape}')
    print(f'device       : {args.device}  gpus={args.gpus}')
    print(f'output dir   : {cli.output_dir}')
    print('=' * 78)

    # Reference statistics: computing them on one sample would make std meaningless.
    ref_idx, _, _ = sample_indices(n_samples, cli.ref_samples, 0, 0, seed=cli.seed)
    ref_set = GOES13Dataset(cli.zarr, ref_idx, preload=False)
    mean_value = float(np.asarray(ref_set.mean).reshape(-1)[0])
    std_value = float(np.asarray(ref_set.std).reshape(-1)[0])
    print(f'normalization: mean={mean_value:.6f} std={std_value:.6f} '
          f'(from {len(ref_idx)} reference samples)')

    single_set = GOES13Dataset(cli.zarr, np.array([cli.index]),
                               mean=ref_set.mean, std=ref_set.std, preload=True)
    # Single sample => must NOT drop the last batch, and must not shuffle.
    single_loader = create_loader(
        single_set, batch_size=1, shuffle=False, is_training=True,
        num_workers=0, pin_memory=True, drop_last=False, persistent_workers=False)

    exp = SingleSampleExperiment(args, dataloaders=(single_loader, single_loader, single_loader))

    print('>' * 35 + ' overfitting ' + '<' * 35)
    exp.train()
    print(f'loss history : {len(exp.loss_history)} epochs, '
          f'first={exp.loss_history[0]:.6f}, last={exp.loss_history[-1]:.6f}')

    # ------------------------------------------------------------ forecast ---
    exp.method.eval()
    batch_x, batch_y = single_set[0]
    batch_x = batch_x.unsqueeze(0).to(torch.device(args.device))
    with torch.no_grad():
        # SimVP.forward truncates when aft < pre, so pred matches the 1 target frame.
        pred = exp.method(batch_x)
    pred = pred.float().cpu().numpy()

    truth = denorm(batch_y.numpy(), mean_value, std_value)
    prediction = denorm(pred, mean_value, std_value)
    condition = denorm(batch_x[0].float().cpu().numpy(), mean_value, std_value)

    print(f'batch shapes : x={tuple(batch_x.shape)} -> pred={tuple(pred.shape)} '
          f'| truth={truth.shape}')

    metrics = scalar_metrics(prediction[0, 0, 0], truth[0, 0])
    print('metrics (de-normalized units):')
    for name, value in metrics.items():
        print(f'  {name:5s}: {value:.6f}')

    # --------------------------------------------------------- artifacts ---
    error = prediction[0, 0, 0] - truth[0, 0, 0]
    np.save(osp.join(cli.output_dir, f'sample_{cli.index}_condition.npy'), condition)
    np.save(osp.join(cli.output_dir, f'sample_{cli.index}_truth.npy'), truth)
    np.save(osp.join(cli.output_dir, f'sample_{cli.index}_pred.npy'), prediction)
    np.save(osp.join(cli.output_dir, f'sample_{cli.index}_error.npy'), error)

    save_panel(
        [condition[0, 0], condition[1, 0], truth[0, 0], prediction[0, 0, 0], error],
        ['condition t-2', 'condition t-1', 'ground truth', 'predicted', 'error (pred - truth)'],
        osp.join(cli.output_dir, f'sample_{cli.index}_comparison.png'),
        vmin=cli.vmin, vmax=cli.vmax)

    emax = float(np.abs(error).max()) or 1.0
    save_panel(
        [error],
        [f'error map, scale [-{emax:.2f}, {emax:.2f}]'],
        osp.join(cli.output_dir, f'sample_{cli.index}_error.png'),
        cmap='coolwarm', vmin=-emax, vmax=emax)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(exp.loss_history) + 1), exp.loss_history, marker='.', linewidth=1)
    ax.set_xlabel('epoch')
    ax.set_ylabel('train_loss (normalized space)')
    ax.set_title(f'Single-sample overfit loss (index {cli.index})')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(osp.join(cli.output_dir, f'sample_{cli.index}_loss_curve.png'), dpi=150)
    plt.close(fig)

    summary = {
        'script': 'tools/train_simvp_goes13_single_sample.py',
        'zarr': cli.zarr,
        'sample_index': cli.index,
        'ref_samples': int(cli.ref_samples),
        'mean': mean_value,
        'std': std_value,
        'args': {k: v for k, v in args.__dict__.items() if isinstance(v, (int, float, str, bool, list))},
        'loss_first': exp.loss_history[0] if exp.loss_history else None,
        'loss_last': exp.loss_history[-1] if exp.loss_history else None,
        'metrics': metrics,
    }
    with open(osp.join(cli.output_dir, f'sample_{cli.index}_summary.json'), 'w') as handle:
        json.dump(summary, handle, indent=2)

    print(f'Artifacts written to {cli.output_dir}')
    print('Interpretation hint: on a correctly wired SimVP this single-sample loss should')
    print('drop by orders of magnitude and the "predicted" panel should closely match')
    print('"ground truth" (small error map). A flat loss or a structureless prediction')
    print('means the model/structure wiring is wrong, not that the data is hard.')


if __name__ == '__main__':
    main()
