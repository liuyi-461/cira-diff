# Satellite Forecasting Lab

本仓库正从 CIRA-Diff working repository 增量升级为 **satellite cloud-image forecasting / satellite nowcasting** 科研实验室。长期问题是：如何利用历史静止气象卫星观测预报未来云场，当前从单通道红外亮温开始。

## 从这里开始

- [Repository audit](docs/project/repository_audit.md) — 当前实现、记录、完成项、计划项和未知项。
- [Current state](docs/project/current_state.md) — Research State 与 Engineering State。
- [Lab architecture](docs/project/architecture.md) — 代码、科研、复现、实验和评价。
- [Research questions](docs/research/questions.md) 与 [hypotheses](docs/research/hypotheses.md)。
- [Dataset system](docs/data/README.md) 与 [evaluation system](docs/evaluation/README.md)。
- [Experiment registry](docs/experiments/registry.md)。
- [Reproduction registry](reproduction/README.md)。
- [CIRA-Diff migration plan](docs/project/migration.md)。

## Project identity

CIRA-Diff 是当前起点，因为它提供了一个具体且可追溯的 satellite-only baseline：两帧历史 GOES-16 ABI Channel 13 图像、一个 10 分钟目标帧，以及到 3 小时的 18 步 autoregressive rollout。这个设计选择是 baseline 事实，不是对卫星预报正确时间建模范式的结论。

Lab 将在明确的数据合同和评价合同下比较：

- 短历史 → 单步预测 → autoregressive rollout；
- 长历史 → 多未来帧序列预测；
- 确定性与概率预报；
- 云场运动、内禀演变、尺度依赖结构和高影响天气型。

## Current status

[FACT] 受保护的 working implementation 仍在 `cira_diff/`，旧训练脚本位于 `scripts/Chase_2025/`。

[RESULT] 现有 knowledge base 记录了远端 CIRA-Diff Zarr 审计和计划中的本地开发子集，但当前仓库没有已验证的本地模型运行或统一 evaluation CLI。

[PLAN] 新代码已在 `src/satforecast/` 下建立 scaffold；在 import、upstream compatibility 和 reproduction behavior 验证前，暂不迁移实现。

## 如何使用 Lab

- 检查当前 baseline：从 `reproduction/cira_diff/` 和 `docs/research/cira_diff.md` 开始。
- 运行 CIRA-Diff：先在实验卡中记录环境、数据路径、config、seed 和 artifact；下面保留的历史命令在当前 checkout 中尚未验证。
- 增加模型：先建立 `reproduction/<model>/` 记录，再定义 data/evaluation adapter；只有 reference behavior 和测试通过后，才 promotion 到 `src/satforecast/`。
- 增加实验：复制 `docs/experiments/templates/experiment_card.md`，在 `docs/experiments/registry.md` 注册 ID，并分开记录 result、interpretation 和 conclusion。
- 评价预报：遵循 `docs/evaluation/evaluation_protocol.md`，记录 lead time、mask、threshold、normalization 和 artifact 路径。
- 不得把 planned baseline、论文指标或生成图像当作已复现的科研结果。

## Existing CIRA-Diff workflow

下面保留原始的 upstream-oriented overview 作为 provenance。在实验卡中记录命令、环境、数据路径和 artifact 前，其中的历史说明都应视为未验证。

---

# 历史上游项目概览（原有内容保留）

以下旧版上游说明仍保留作为 provenance。新增 canonical 文档统一使用中文；其中的英文原文暂不删除，以避免改变历史记录。

主要联系人：Randy Chase
邮箱：dopplerchase12 'at' gmail.com

## Introduction 

![Alt Text](./aux/movies/output.gif)
This repository is to hold the code for the diffusion model efforts at CIRA-CSU. The first couple projects for us are to do are conditional diffusion models to do image2image translation using satellite data. Specifcally, we began by forecasting GOES IR brightness temperatures out the 3 hours and the conversion of GOES data to passive microwave observations. The paper is titled: [Score based diffusion nowcasting of GOES imagery](), and an example forecast is included in the gif above. More projects to come.

## Background and our journey
When we first stared on learning how to encorperate diffusion models into our workflow, we started with [this](https://huggingface.co/docs/diffusers/en/tutorials/basic_training) example from HuggingFace that trains an <i> unconditional </i> diffusion model that generates pictures of butterflies using Denoising Diffusion Probabilistic Models ([DDPM](https://arxiv.org/abs/2006.11239)). This was a useful place to start, but with most meteorology/atmos tasks, conditional modeling we find is much more useful. To include a condition we found it useful to concatenate your condition alongside the noisy dimension (see our paper for discussion). This worked following DDPM, but with the build in DDPM sampler, it was requiring something like 1000 neural network calls to get decent data. This was just too computationally expensive to get into any operational environment. 

We then moved on to following the work out of Google and NVIDIA, where they both closely follow the [Karras et al. (2022)](https://arxiv.org/abs/2206.00364) titled: Elucidating the Design Space of Diffusion-Based Generative Models (hereafter EDM). The key advantages we found of following the EDM approach over DDPM:

1) Calls to the network are less than 100 for <i> good </i> performance 
2) training was relatively easy and stable (just long....)
3) more advancement coming out of the NVIDIA group ([Karras et al. 2024a](https://arxiv.org/abs/2312.02696),[Karras et al. 2024b](https://arxiv.org/abs/2406.02507)). 
4) NVIDIA had the code already implemented in MODULUS and used it for CorrDiff/StormCast 

Our implementation of the code comes directly out of the [original repo](https://github.com/NVlabs/edm), not MODULUS because of the bloat with modulus (i.e., we don't need all their functionality). Turns out though that MODULUS also took the main code from EDM and wrapped it with the rest of their repo. We only grab the train/generate code out of the EDM repo, and then we leverage HuggingFace's [diffusers]() as our architecture hub so we could play around with various <i> drop in </i> architectures. 

Eventually, this repo will be more generalized, but I have run out of time to do so. So for now, you will have to dig into the scripts I used to train the models for the papers located in ``./scripts/Chase2025/``.

## Getting Started (installing things)
1. Setup a Python installation on the machine you are using. If you already have conda/mamba move one to 2.  

   I recommend installing [miniforge](https://github.com/conda-forge/miniforge). Inside miniforge is Mamba, which tends to solve environments more quickly than conda and miniconda. Also anaconda has a new license out there that charges for things. Mamba is more of an open-source and free version of conda. 

2. Install a env

   We are including an environment.yml file here, but given the variety of GPUs out there, folks will probably have a challenge here installing the right torch version for their GPUs. What I am going to suggest is to do the install in steps. First make a new env: 

   ``mamba create -n cira-diff``

   activate it 

   ``mamba activate cira-diff``

   Then install pytorch first: 
   
   ``mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia``

   Note this is how I installed pytorch for the CUDA my GPUs have. If you dont have CUDA 12, change this to one of the 11.8s or something. You can see which CUDA is compiled by running `nvidia-smi` on a node where GPUs are connected. Alternatively, if you are using GH200s, you will need to use docker and the precompiled pytorch they give you. Example [here](https://dopplerchase-ai2es-schooner-hpc.readthedocs.io/en/latest/cira.html#gh200-how-to).

   Next up, install diffusers, transformers and accelerate. If you don't want to use diffuser models, you could skip this, but know the code will break because we import it later. We also install some other common packages here too: 

   ``pip install diffusers["torch"] transformers accelerate matplotlib tensorboard py3nvml build zarr`` 

5. Install local repo 

   `` pip install . `` 
   
## Data Prep

If you want to make this repo work for your dataset, the easiest implementation would be to adapt your training dataset to fit with the code. Our training dataset was of the shape ``(generation_images, condition images)`` where ``generation images`` was ``[n_samples,generation_channels,nx,ny]`` and condition images was ``[n_samples,condition_channels,nx,ny]``. We made this dataset by: 

1. loading a bunch of files 
2. slicing them down to a reasonable size (256 by 256, and the literature suggests smaller is better). 
3. find mean and std of the data
4. normalize all data to have mean=0 and std=1 
5. save out a zarr file that will return a the tuple: ``(generation_images, condition images, nx, ny)``

If you want to use our data for now, please check out our online data repo on [dryad]()

## Training

Now that you have a dataset ready, go ahead and train one of the models. Be sure to update the file paths in the top of the training scripts to what you need. Example call 

`` accelerate launch train_edm_Chase2025.py `` 

## Generation 

An example notebook of how to run a bunch of forecasts, say on the validation or test set is also included in the scripts folder. The name is `` Run_Forecasts_Chase2025.ipynb``. Feel free to grab everything out of the notebook and put it into a script. 


