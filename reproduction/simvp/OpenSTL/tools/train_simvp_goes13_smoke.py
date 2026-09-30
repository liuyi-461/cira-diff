"""Small-sample (smoke) training script validating the full OpenSTL pipeline.

Purpose
-------
Drive the whole SimVP training + testing pipeline end-to-end (data -> loaders ->
Lightning training loop -> validation -> test metrics -> saved artifacts) on a
small subsample of the EDM GOES-16 ABI channel-13 zarr store, so that the
pipeline can be verified before spending GPU time on the full dataset.

Why a standalone script instead of ``tools/train.py -d goes13``
---------------------------------------------------------------
``tools/train.py`` resolves data through ``get_dataset()``, which requires the
dataset to be registered in ``openstl.datasets.dataset_constant`` /
``load_data`` / the argparse ``choices`` of ``--dataname``. Registering it would
mean editing several upstream files. Instead we use the officially supported
escape hatch ``BaseExperiment(args, dataloaders=...)`` (the same mechanism the
custom-data tutorial uses), together with new files only:

    openstl/datasets/dataloader_goes13.py   # dataset adapter (new)
    configs/goes13/simvp/SimVP_gSTA.py      # hyper-parameters (new)
    tools/train_simvp_goes13_smoke.py       # this entrypoint (new)

Documented deviations vs upstream:
  D1) ``BaseExperiment._init_trainer`` hardcodes ``accelerator='gpu'``. We
      subclass it as ``SmokeExperiment`` so a CPU-only machine can still run the
      smoke test; nothing else changes.
  D2) The zarr store has no ``_ARRAY_DIMENSIONS`` metadata, so it is read with
      the plain zarr API (``xarray.open_zarr`` raises KeyError).
  D3) Normalization statistics are computed on the sampled subset, not on the
      full training split (acceptable for a pipeline check, not for results).

Usage
-----
    python tools/train_simvp_goes13_smoke.py \
        --zarr /path/to/satcast/edm_GOES_ch13_train_dataset.zarr \
        --res_dir /abs/path/to/OpenSTL/work_dirs \
        --ex_name simvp_goes13_smoke \
        --n-train 128 --n-val 32 --n-test 32 --epochs 3 --batch-size 8

On a scheduler use ``sbatch test_dl/test_dl.slurm`` (absolute paths baked in).

``--res_dir`` / ``--ex_name`` are upstream options and therefore spelled with an
underscore; always pass absolute paths, because the defaults are relative and the
landing directory would otherwise depend on the job's working directory.

Outputs land in ``<res_dir>/<ex_name>/`` (checkpoints, ``saved/*.npy``,
``model_param.json``). The directories are created automatically when training
starts; nothing has to be pre-created.

Dependencies: torch, lightning, timm, fvcore, zarr, numpy, opencv-python
(``timm`` and ``fvcore`` are already required by OpenSTL itself).
"""

import os
import os.path as osp
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import torch
import zarr
from lightning import Trainer

from openstl.api import BaseExperiment
from openstl.datasets.dataloader_goes13 import (
    AFT_SEQ_LENGTH, CHANNELS, DATA_NAME, IMAGE_SIZE, PRE_SEQ_LENGTH,
    load_data, sample_indices,
)
from openstl.utils import create_parser, default_parser, get_dist_info, load_config, update_config

DEFAULT_ZARR = '/data1/satcast/edm_GOES_ch13_train_dataset.zarr'
HERE = osp.dirname(osp.abspath(__file__))
DEFAULT_CONFIG = osp.join(HERE, '..', 'configs', 'goes13', 'simvp', 'SimVP_gSTA.py')


class SmokeExperiment(BaseExperiment):
    """BaseExperiment with a single documented deviation (D1).

    Upstream hardcodes ``accelerator='gpu'``; here the accelerator follows the
    machine so the pipeline can be smoke-tested on CPU as well.
    """

    def _init_trainer(self, args, callbacks, strategy):
        use_gpu = torch.cuda.is_available() and str(args.device).startswith('cuda')
        return Trainer(
            devices=args.gpus if use_gpu else 1,
            max_epochs=args.epoch,
            strategy=strategy,
            accelerator='gpu' if use_gpu else 'cpu',
            callbacks=callbacks,
        )


def make_cli_parser():
    parser = create_parser()
    parser.description = 'SimVP small-sample smoke training on EDM GOES-13 data'
    # NOTE: we deliberately do not rely on argparse choices for --dataname, so the
    # newly added dataset name is injected programmatically instead.
    parser.add_argument('--zarr', default=DEFAULT_ZARR, type=str,
                        help='Path to edm_GOES_ch13_train_dataset.zarr')
    parser.add_argument('--config', default=DEFAULT_CONFIG, type=str,
                        help='OpenSTL-style hyper-parameter file')
    parser.add_argument('--n-train', type=int, default=128, help='Training subsample size')
    parser.add_argument('--n-val', type=int, default=32, help='Validation subsample size')
    parser.add_argument('--n-test', type=int, default=32, help='Test subsample size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Train / val batch size')
    parser.add_argument('--model-type', default='gSTA', type=str,
                        help='SimVP hidden translator backbone')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--no-preload', action='store_true', default=False,
                        help='Read from zarr on the fly instead of caching the subset')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Force CPU even when CUDA is available')
    parser.add_argument('--display-method-info', action='store_true', default=False,
                        help='Print model summary / FLOPs (needs fvcore)')
    # NOTE: '--lr' is already registered by create_parser(); re-adding it raises
    # "argparse.ArgumentError: conflicting option string: --lr".
    # Also, several flags added here share their dest with an upstream flag
    # (--model_type / --batch_size / --num_workers). argparse keeps only the FIRST
    # registered default per dest, so our defaults were silently ignored
    # (model_type=None -> MetaBlock NotImplementedError; batch_size=16; num_workers=4).
    # set_defaults() patches the default of every action sharing the dest, so the
    # intended defaults take effect while keeping both spellings usable.
    parser.set_defaults(lr=1e-3, model_type='gSTA', batch_size=8, num_workers=2)
    return parser


def build_args(cli):
    """Compose the OpenSTL args namespace: defaults <- config file <- CLI."""
    args = create_parser().parse_args([])

    # 1) hyper-parameters from the config file (model + optimizer/scheduler)
    cfg = load_config(cli.config)
    update_config(args.__dict__, cfg, exclude_keys=['method'])

    # 2) anything the config file did not provide falls back to a documented default
    for key, value in default_parser().items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if not hasattr(args, 'ckpt_path'):
        args.ckpt_path = None

    # 3) task definition + CLI overrides win over everything above
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
        'batch_size': cli.batch_size,
        'val_batch_size': cli.batch_size,
        'num_workers': cli.num_workers,
        'lr': cli.lr,
        'model_type': cli.model_type,
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


def main():
    cli = make_cli_parser().parse_args()

    store = zarr.open(cli.zarr, mode='r')
    n_samples = store['output_images'].shape[0]
    train_idx, val_idx, test_idx = sample_indices(
        n_samples, cli.n_train, cli.n_val, cli.n_test, seed=cli.seed)

    args = build_args(cli)

    print('=' * 78)
    print('SimVP smoke run — EDM GOES-16 ABI ch13 (2 -> 1 frames)')
    print('=' * 78)
    print(f'zarr store      : {cli.zarr}  ({n_samples} samples available)')
    print(f'config file     : {cli.config}')
    print(f'subsample       : train={len(train_idx)} val={len(val_idx)} '
          f'test={len(test_idx)} (seed={cli.seed})')
    print(f'method/model    : {args.method} / model_type={args.model_type}')
    print(f'in_shape        : {args.in_shape} '
          f'(pre={args.pre_seq_length}, aft={args.aft_seq_length})')
    print(f'epochs / bs / lr: {args.epoch} / {args.batch_size} / {args.lr}')
    print(f'device          : {args.device}  gpus={args.gpus}')
    print(f'out dir         : {osp.join(args.res_dir, args.ex_name)}')
    print('=' * 78)

    if len(train_idx) < args.batch_size:
        raise ValueError(
            f'train subsample ({len(train_idx)}) is smaller than batch_size '
            f'({args.batch_size}); the train loader uses drop_last=True.')

    # --- DBG-007 护栏：全量 preload 会把整个子集读进内存 ---
    # 每样本 ≈ 3 帧 × 256×256 × 4 字节 ≈ 0.79 MB；全量 35595 样本 ≈ 28 GB。
    # 全量训练务必加 --no-preload（逐条读 zarr），否则极易 OOM。
    if (not cli.no_preload) and (len(train_idx) + len(val_idx) + len(test_idx)) > 4000:
        _total = len(train_idx) + len(val_idx) + len(test_idx)
        _gb = _total * 3 * IMAGE_SIZE * IMAGE_SIZE * 4 / 1e9
        print(f'[DBG-007][WARN] preload=True 且总样本={_total} 会一次性读入约 '
              f'{_gb:.1f} GB 内存；全量训练请加 --no-preload，否则极易 OOM。')

    dataloader_train, dataloader_vali, dataloader_test = load_data(
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        data_root=cli.zarr,
        indices_train=train_idx,
        indices_val=val_idx,
        indices_test=test_idx,
        num_workers=args.num_workers,
        preload=not cli.no_preload,
        distributed=args.dist,
        use_prefetcher=args.use_prefetcher,
        drop_last=args.drop_last,
    )
    print(f'normalization   : mean={dataloader_train.dataset.mean.item():.6f} '
          f'std={dataloader_train.dataset.std.item():.6f} (from the train subsample)')
    batch_x, batch_y = next(iter(dataloader_train))
    print(f'batch shapes    : x={tuple(batch_x.shape)} y={tuple(batch_y.shape)}')

    exp = SmokeExperiment(
        args, dataloaders=(dataloader_train, dataloader_vali, dataloader_test))

    print('>' * 35 + ' training ' + '<' * 35)
    exp.train()

    rank, _ = get_dist_info()
    if rank == 0:
        print('>' * 35 + ' testing  ' + '<' * 35)
    exp.test()

    if rank == 0:
        print('Smoke run finished. Artifacts dir: '
              f'{osp.join(exp.save_dir)}')


if __name__ == '__main__':
    main()
