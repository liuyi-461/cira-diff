"""SimVP 全量训练（baseline 对齐版）—— GOES-16 ABI ch13，2 帧 -> 1 帧单步。

与 ``train_simvp_goes13_smoke.py``（EXP-002）的区别
--------------------------------------------------
EXP-002 的 smoke 脚本有两个与 baseline 不兼容的设计，本脚本全部改掉：

  1. **三个 split 从同一份 train zarr 抽样** -> 改为 train/validation/test 各读一份 zarr，
     validation/test 是真正的 held-out（Chase 2025 官方 split，各 1024 条）。
  2. **归一化用子集自算的 mean/std**（D3）-> 改为**不做任何归一化**，直接吃 zarr 原值。
     zarr 内已是归一化空间（mean=0, std=1，见 ``satcast/simple_code.md``），
     物理常数 ``MEAN_K/STD_K`` 只用于把指标换算回亮温(K)，由 ``dataloader_satcast`` 提供。

这样训出来的 checkpoint 可以直接用统一评估脚本按 baseline 协议评估：::

    python tools/eval_simvp_baseline.py --ckpt <...>/best.ckpt --norm-mean 0 --norm-std 1

（对照：EXP-002 的老 ckpt 训练时用了子集常数 0.093006/0.865920，评估时必须还原。）

超参来源
--------
全部来自 ``configs/satcast/SimVP.py``（唯一来源）。注意上游 ``update_config`` 会静默
丢弃"argparse 默认值为真值"的 key（``openstl/utils/main_utils.py:146``），所以本脚本
把配置文件的**所有** key 放进 ``exclude_keys``，确保配置真的生效。

用法
----
全量训练（100 epoch，由 config 决定）::

    python tools/train_simvp_satcast.py \
        --res_dir /abs/path/OpenSTL/work_dirs --ex_name satcast_simvp_full

秒级冒烟（确认代码+数据通路，产物写到独立 ex_name）::

    python tools/train_simvp_satcast.py --cpu --epochs 1 \
        --limit-train 16 --limit-val 8 --limit-test 8 \
        --ex_name satcast_simvp_smoke

Outputs: ``<res_dir>/<ex_name>/`` 下有 checkpoints/（含 best.ckpt）、model_param.json。
**权威指标请用 ``tools/eval_simvp_baseline.py``**，不要用本脚本的 exp.test()
（OpenSTL 的 test 跑在最终权重上，且 metrics.py 的空间求和口径会放大 65536 倍）。

Dependencies: torch, lightning, timm, fvcore, zarr, numpy
"""

import os.path as osp
import sys
import warnings

warnings.filterwarnings('ignore')

# 自举：直接 `python tools/train_simvp_satcast.py` 时 sys.path[0] 是 tools/，
# 找不到顶层 openstl 包。这里显式把仓库根加进来，使脚本不依赖外部 PYTHONPATH。
_HERE = osp.dirname(osp.abspath(__file__))
_OPENSTL_DIR = osp.dirname(_HERE)
if _OPENSTL_DIR not in sys.path:
    sys.path.insert(0, _OPENSTL_DIR)

import torch
from lightning import Trainer

from openstl.api import BaseExperiment
from openstl.datasets.dataloader_satcast import MEAN_K, STD_K, load_data
from openstl.utils import (create_parser, default_parser, get_dist_info,
                           load_config, update_config)

DATA_NAME = 'satcast_ch13'
PRE_SEQ_LENGTH = 2
AFT_SEQ_LENGTH = 1
IMAGE_SIZE = 256
CHANNELS = 1

HERE = _HERE
OPENSTL_DIR = _OPENSTL_DIR
DEFAULT_CONFIG = osp.join(HERE, '..', 'configs', 'satcast', 'SimVP.py')
DEFAULT_DATA_ROOT = '/home/group1/26fall_aiclass/yr/data1/data1/satcast'


class SatcastExperiment(BaseExperiment):
    """BaseExperiment + 偏离 D1：accelerator 跟随机器，便于无 GPU 时冒烟。"""

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
    parser.description = 'SimVP full training on Satcast GOES-13 (baseline aligned, 2->1)'
    parser.add_argument('--config', default=DEFAULT_CONFIG, type=str,
                        help='超参唯一来源（configs/satcast/SimVP.py）')
    parser.add_argument('--satcast-root', default=None, type=str,
                        help='覆盖 config 里的 data_root')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖 config 里的 epoch（冒烟用 1）')
    parser.add_argument('--limit-train', type=int, default=None,
                        help='只取前 N 条训练样本（冒烟用；正式训练保持 None）')
    parser.add_argument('--limit-val', type=int, default=None)
    parser.add_argument('--limit-test', type=int, default=None)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--run-test', action='store_true', default=False,
                        help='训练后跑 exp.test()；权威指标请用 eval_simvp_baseline.py')
    parser.add_argument('--display-method-info', action='store_true', default=False)
    # 与 smoke 脚本同理：--model_type 与上游共享 dest，用 set_defaults 修正默认
    parser.set_defaults(model_type='gSTA')
    return parser


def build_args(cli):
    """Compose the OpenSTL args namespace: 配置文件为唯一超参来源。"""
    args = create_parser().parse_args([])

    # 1) 超参全部来自配置文件。
    #    exclude_keys=全部 key，是为了绕开 update_config 的静默丢弃行为
    #    （openstl/utils/main_utils.py:146）——否则 batch_size/num_workers/lr/epoch
    #    等"argparse 默认值为真值"的 key 会被 CLI/argparse 默认值盖掉，配置形同虚设。
    cfg = load_config(cli.config)
    update_config(args.__dict__, cfg, exclude_keys=list(cfg.keys()))

    # 2) 配置未提供的项回落到 documented defaults
    for key, value in default_parser().items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if not hasattr(args, 'ckpt_path'):
        args.ckpt_path = None

    # 3) 任务定义 + 路径（这些 key 不在配置里，可安全用 CLI 覆盖）
    overrides = {
        'method': 'SimVP',
        'dataname': 'satcast',
        'data_name': DATA_NAME,
        'model_type': cli.model_type,
        'in_shape': [PRE_SEQ_LENGTH, CHANNELS, IMAGE_SIZE, IMAGE_SIZE],
        'pre_seq_length': PRE_SEQ_LENGTH,
        'aft_seq_length': AFT_SEQ_LENGTH,
        'total_length': PRE_SEQ_LENGTH + AFT_SEQ_LENGTH,
        'metrics': ['mse', 'mae'],
        'metric_for_bestckpt': 'val_loss',
        'device': 'cpu' if cli.cpu or not torch.cuda.is_available() else 'cuda',
        'gpus': list(cli.gpus),
        'ex_name': cli.ex_name,
        'res_dir': cli.res_dir,
        'seed': cli.seed,
        'dist': cli.dist,
        'use_prefetcher': cli.use_prefetcher,
        'drop_last': False,
        'no_display_method_info': not cli.display_method_info,
        # 续跑：上游 --ckpt_path 会交给 Trainer.fit(ckpt_path=...)，
        # 恢复 epoch / optimizer / scheduler 状态。
        'ckpt_path': getattr(cli, 'ckpt_path', None),
    }
    for key, value in overrides.items():
        setattr(args, key, value)

    if cli.epochs is not None:
        args.epoch = int(cli.epochs)
    if cli.satcast_root:
        args.data_root = cli.satcast_root

    # DBG-004 护栏：上游 res_dir / ex_name 的默认值是相对路径（'work_dirs' / 'Debug'），
    # 产物落点会随作业工作目录漂移。这里把 res_dir 统一解析成绝对路径，
    # 并对仍是默认值的 ex_name 给出警告。
    if not osp.isabs(args.res_dir):
        print(f'[WARN] res_dir 是相对路径 "{args.res_dir}"，已解析为 '
              f'"{osp.join(OPENSTL_DIR, args.res_dir)}"（DBG-004）')
        args.res_dir = osp.join(OPENSTL_DIR, args.res_dir)
    if args.ex_name == 'Debug':
        print('[WARN] ex_name 仍是上游默认值 "Debug"，正式训练请显式传 --ex_name')
    return args


def main():
    cli = make_cli_parser().parse_args()
    args = build_args(cli)

    print('=' * 78)
    print('SimVP 全量训练（baseline 对齐）— EDM GOES-16 ABI ch13 (2 -> 1)')
    print('=' * 78)
    print(f'config file     : {cli.config}')
    print(f'data_root       : {args.data_root}')
    print(f'method/model    : {args.method} / {args.model_type}')
    print(f'in_shape        : {args.in_shape} '
          f'(pre={args.pre_seq_length}, aft={args.aft_seq_length})')
    print(f'epochs / bs / lr: {args.epoch} / {args.batch_size} / {args.lr}')
    print(f'num_workers     : {args.num_workers}')
    print(f'device          : {args.device}  gpus={args.gpus}')
    print(f'out dir         : {osp.join(args.res_dir, args.ex_name)}')
    print('-' * 78)
    print('归一化：**不做**（zarr 已是 mean=0/std=1 的归一化空间）；')
    print(f'物理常数 MEAN_K={MEAN_K:.4f} K, STD_K={STD_K:.4f} K，仅用于把指标换算回亮温。')
    print('=> 评估该 checkpoint 时用：--norm-mean 0 --norm-std 1')
    print('=' * 78)

    dataloader_train, dataloader_vali, dataloader_test = load_data(
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        data_root=args.data_root,
        num_workers=args.num_workers,
        pre_seq_length=args.pre_seq_length,
        aft_seq_length=args.aft_seq_length,
        distributed=args.dist,
        use_prefetcher=args.use_prefetcher,
        drop_last=args.drop_last,
        limit_train=cli.limit_train,
        limit_val=cli.limit_val,
        limit_test=cli.limit_test,
    )

    batch_x, batch_y = next(iter(dataloader_train))
    print(f'batch shapes    : x={tuple(batch_x.shape)} y={tuple(batch_y.shape)}')

    exp = SatcastExperiment(
        args, dataloaders=(dataloader_train, dataloader_vali, dataloader_test))

    print('>' * 35 + ' training ' + '<' * 35)
    exp.train()

    if cli.run_test:
        rank, _ = get_dist_info()
        if rank == 0:
            print('[NOTE] exp.test() 的数字**不是**权威指标：它跑在最终权重上，')
            print('       且 metrics.py 的空间求和口径会放大 65536 倍。')
            print('       请用 tools/eval_simvp_baseline.py 评估 best.ckpt。')
            print('>' * 35 + ' testing  ' + '<' * 35)
        exp.test()

    print('Training finished. Artifacts dir: ' + str(exp.save_dir))
    print('best.ckpt -> ' + osp.join(str(exp.save_dir), 'checkpoints', 'best.ckpt'))


if __name__ == '__main__':
    main()
