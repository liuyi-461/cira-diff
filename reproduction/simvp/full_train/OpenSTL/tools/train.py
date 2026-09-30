# Copyright (c) CAIRI AI Lab. All rights reserved

import os.path as osp
import warnings
warnings.filterwarnings('ignore')

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, get_dist_info, load_config,
                           update_config)


if __name__ == '__main__':
    # ---- 本地新增（非上游行为）----
    # 开启 TF32 的 matmul 路径。torch 2.6 的默认是：
    #   cudnn.allow_tf32 = True（卷积**已经**在用 TF32）
    #   float32_matmul_precision = 'highest'（matmul **没用** TF32）
    # SimVP 的 gSTA backbone 在 MetaBlock 的 MLP 里有大量 matmul，所以这一项
    # 是实打实的提速，且这是 PyTorch 官方对 Ampere/Ada 卡的推荐设置
    # （单样本测试的 stderr 里 Lightning 也明确提示了这一条）。
    # 代价：TF32 尾数是 10 bit，数值精度略降，对训练/推理精度影响可忽略。
    import torch
    torch.set_float32_matmul_precision('high')

    args = create_parser().parse_args()
    config = args.__dict__

    cfg_path = osp.join('./configs', args.dataname, f'{args.method}.py') \
        if args.config_file is None else args.config_file
    if args.overwrite:
        config = update_config(config, load_config(cfg_path),
                               exclude_keys=['method'])
    else:
        loaded_cfg = load_config(cfg_path)
        # ---- 本地改动（非上游行为）----
        # 上游 update_config（openstl/utils/main_utils.py）的行为是：若某 key 的
        # argparse 默认值是**真值**且不在 exclude_keys 里，配置文件中的值会被
        # **静默丢弃**。这会踩中 satcast 需要的 batch_size(默认16)、
        # data_root('./data')、num_workers(4)、min_lr(1e-6)、opt('adam')，
        # 使 configs/satcast/SimVP.py 里写的值不生效。
        #
        # 为了让实验配置只有**一个**来源（就是那个配置文件），这里把 satcast
        # 关心的这几项也放进 exclude_keys —— 含义变为"这些 key 一律以配置文件
        # 为准，命令行不再覆盖"。上游原有的四项保持不变。
        exclude_keys = ['method', 'val_batch_size', 'drop_path', 'warmup_epoch',
                        'batch_size', 'data_root', 'num_workers', 'min_lr', 'opt']
        config = update_config(config, loaded_cfg, exclude_keys=exclude_keys)
        default_values = default_parser()
        for attribute in default_values.keys():
            if config[attribute] is None:
                config[attribute] = default_values[attribute]

    print('>'*35 + ' training ' + '<'*35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    exp.train()

    if rank == 0:
        print('>'*35 + ' testing  ' + '<'*35)
    mse = exp.test()