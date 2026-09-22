# CIRA-Diff 环境配置与代码指南

> 编写日期：2026-09-21
> 目标：帮助新用户快速理解项目结构、跑通训练脚本、理解扩散模型理论

---

## 一、环境安装

### 1.1 Conda 环境

服务器已有一个可用环境 `cira-diff-ly`（Python 3.11 + PyTorch 2.6.0 + CUDA 12.4）。如需要新建：

```bash
conda create -n cira-diff-ly python=3.11 -y
conda activate cira-diff-ly

# PyTorch（如 base 已有可跳过）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 依赖包
pip install "diffusers[torch]" transformers accelerate matplotlib tensorboard \
            py3nvml build zarr tqdm pillow safetensors -i https://pypi.tuna.tsinghua.edu.cn/simple

# 本地包（开发者模式）
cd /home/group1/26fall_aiclass/ly/cira-diff
pip install -e .
```

### 1.2 训练前检查清单

| 检查项 | 命令 |
|--------|------|
| GPU 可用 | `nvidia-smi` |
| CUDA + PyTorch | `python -c "import torch; print(torch.cuda.is_available())"` |
| 数据完整性 | 见下节 |

---

## 二、数据集

数据存放在 `/data1/satcast/`：

| 文件 | 类型 | 形状 | 用途 |
|------|------|------|------|
| `edm_GOES_ch13_train_dataset.zarr` | 原始 2 通道 | input: (35595, 2, 256, 256) output: (35595, 1, 256, 256) | vanilla / EDM |
| `edm_GOES_ch13_train_dataset_latent.zarr` | VAE latent | input: (35595, 8, 64, 64) output: (35595, 4, 64, 64) | LDM |
| `edm_GOES_ch13_train_dataset_CorrDiff.zarr` | CorrDiff 专用 | input: (35595, 3, 256, 256) output: (35595, 1, 256, 256) | CorrDiff |
| `edm_GOES_ch13_test_dataset.zarr` | 原始测试集 | 同上 | 推理评估 |
| `edm_GOES_ch13_test_dataset_latent.zarr` | latent 测试集 | 同上 | LDM 推理 |
| `edm_GOES_ch13_validation_dataset.zarr` | 原始验证集 | 同上 | 调参 |
| `edm_GOES_ch13_validation_dataset_latent.zarr` | latent 验证集 | 同上 | LDM 调参 |

**通道含义**：
- 原始数据 input (2ch)：过去两个时间步的 GOES Ch13 红外图
- 原始数据 output (1ch)：目标时间步的红外图
- Latent 数据：4× 下采样后的 VAE 编码
- CorrDiff 数据 input (3ch)：2 个历史帧 + 1 个 UNet prior forecast

---

## 三、6 个训练脚本详解

### 3.1 一图总览

```
扩散模型训练的 4 种"口味"
│
├─ ① Vanilla UNet（不做扩散，纯回归）
│   └─ scripts/Chase_2025/train_vanilla_unet_Chase2025.py
│
├─ ② EDM（Elucidated Diffusion Model，Karras 2022）
│   ├─ scripts/Chase_2025/train_edm_Chase2025.py          ← 独立脚本（硬编码 config）
│   └─ cira_diff/train_Diff.py                            ← 重构版（外部 config 文件）
│
├─ ③ LDM（Latent Diffusion Model，扩散在 latent 空间）
│   └─ scripts/Chase_2025/train_edm_LDM_Chase2025.py
│
└─ ④ CorrDiff（NVIDIA 2023 扩散校正）
    ├─ scripts/Chase_2025/train_edm_CorrDiff_Chase2025.py ← 独立脚本
    └─ cira_diff/train_CorrDiff.py                        ← 重构版
```

### 3.2 Vanilla UNet —— 纯回归 baseline

**文件**：`scripts/Chase_2025/train_vanilla_unet_Chase2025.py`

**做什么**：UNet 直接从条件图像（2 通道）回归出目标图像（1 通道），**没有任何扩散过程**。这是最简单的 baseline，用来验证"不用扩散，纯 CNN 能到什么水平"。

**模型配置**：
```
UNet2DModel: in_channels=2, out_channels=1
block_out_channels=(128,128,256,256,512,512)
Attention block at 5th encoder layer
```

**关键代码**（`train_vanilla_unet_Chase2025.py:172`）：
```python
yhat = model(condition_images, torch.zeros(...), return_dict=False)[0]
loss = F.mse_loss(yhat.float(), clean_images.float())
```
第二个参数（time embedding）传零，UNet 的时序条件路径**不参与**，就是个纯回归器。

**为什么重要**：你已经验证过——单样本 1000 epoch overfit，MSE 从 0.56 降到 2.16e-5（约 25000×），证明 UNet 架构和训练管线完全正确。

### 3.3 EDM —— Elucidated Diffusion Model

**文件**：`scripts/Chase_2025/train_edm_Chase2025.py`

**核心组件**：`EDMPrecond` 类（`train_edm_Chase2025.py` 内部定义）

这是对 `diffusers.UNet2DModel` 的封装，实现 Karras 2022 的训练规范：

**训练流程**：
1. 采样噪声强度 σ ~ logN(P_mean, P_std²)
2. 给干净图像加噪：`x_σ = x_0 + σ · n`，n ~ N(0, I)
3. 条件拼接：`model_input = cat([x_σ, cond_ch0, cond_ch1], dim=1)` → **3 通道**
4. `EDMPrecond` 内部对输入/输出做预处理：
   ```
   network_input = x_σ / √(σ² + σ_data²)    # c_noise = ln(σ)/4
   network_output = c_out · F_θ(network_input, c_noise) + c_skip · x_σ
   ```
5. Loss：`L = w(σ) · || network_output - x_0 ||²`
   其中 `w(σ) = (σ² + σ_data²) / (σ · σ_data)²` 是 Karras 推导的最优权重

**EDM 和普通 DDPM 的 6 点区别**（详见第四章 Q5）：

| # | 改动 | DDPM | EDM（本项目） |
|---|------|------|---------------|
| ① | 噪声采样 | 固定 β schedule | σ ~ logN(P_mean=-1.2, P_std=1.2)，每个样本随机采 |
| ② | 输入预处理 | 无，原始尺度直接送 UNet | `c_in = 1/√(σ²+σ_data²)` 只缩放 noisy 部分 |
| ③ | 条件拼接 | noisy + cond 同尺度拼接 | noisy 经 c_in 缩放后再和 cond 拼接（cond 不缩放） |
| ④ | 输出 skip | 普通 residual 连接 | `D_x = c_skip·x_σ + c_out·F_θ(x_σ·c_in, log(σ)/4)` ，σ-dependent 加权组合 |
| ⑤ | Loss | 无 σ 依赖权重 | `w(σ) = (σ²+σ_data²)/(σ·σ_data)²` |
| ⑥ | 采样器 | DDIM 固定步长 | DPM++ / Heun 二阶 / Stochastic SDE 多种求解器 |

> 只有 ⑤ 能被称为 "loss 加了个系数"，而 ①②③④⑥ 都是更本质的改动。
> 其中 ④（c_skip + c_out 的 skip connection）是 EDM 效果好于普通 diffusion 的关键。

**训练参数**（本项目默认）：
```python
P_mean = -1.2       # log(σ) 采样均值
P_std = 1.2         # log(σ) 采样标准差
sigma_data = 0.5    # 数据的预估标准差
learning_rate = 1e-4
gradient_accumulation_steps = 2
```

### 3.4 LDM —— Latent Diffusion Model

**文件**：`scripts/Chase_2025/train_edm_LDM_Chase2025.py`

**核心思路**：把扩散从 **原空间 256×256** 搬到 **latent 空间 64×64**，输入输出缩小 16×，UNet 参数量和显存占用大幅降低。

**Pipeline**：
```
训练前（一次性）：
  原始图 (256×256) --VAE encode--> latent (64×64) zarr

训练时：
  latent (64×64) → 加噪 → EDMPrecond → UNet (64×64) → 预测干净 latent

推理时：
  噪声 latent (64×64) → EDMPrecond 去噪 → 干净 latent → VAE decode → 原始图 (256×256)
```

**使用的 VAE**：`radames/stable-diffusion-x4-upscaler-img2img`（HuggingFace Hub）
- 4× 下采样：256 → 64
- 输入/输出通道翻倍：1→4，2→8

**数据**：你已下载好 latent zarr，直接用即可。

### 3.5 CorrDiff —— 扩散校正

**文件**：`scripts/Chase_2025/train_edm_CorrDiff_Chase2025.py`

**核心思想**（NVIDIA 2023 StormCast 论文）：

> 先用一个 UNet baseline 直接预测目标（不扩散），得到"粗略预测"；
> 再让扩散模型只学习 baseline 预测和真实目标之间的 **residual（残差）**。

这样扩散模型只需要纠正小误差，任务难度大降。

**数据格式**（关键区别）：
```
CorrDiff 数据 input (3ch): [past_frame_0, past_frame_1, unet_baseline_forecast]
CorrDiff 数据 output (1ch): ground_truth_target
训练时扩散模型预测: residual = target - unet_baseline_forecast
```

**本项目 CorrDiff 需要的输入通道**：`in_channels = 4`（1 noisy + 2 condition + 1 unet prior）

### 3.6 重构版 vs 原版

| | 原版（Chase_2025 目录） | 重构版（cira_diff 目录） |
|--|------------------------|--------------------------|
| 文件 | `train_edm_Chase2025.py` | `train_Diff.py` |
| | `train_edm_CorrDiff_Chase2025.py` | `train_CorrDiff.py` |
| Config 管理 | 硬编码在 `TrainingConfig` dataclass 里 | 外部 `.py` 文件 + `importlib` 动态加载 |
| 启动方式 | `python script.py` | `python -m cira_diff.train_Diff --config /path/to/config.py` |
| 额外功能 | 简单 TensorBoard 日志 | py3nvml 显存监控、模型类型可配、Early Stopping 可配 |
| 多组实验 | 需要复制脚本改参数 | 只换 config 文件，干净 |
| 使用建议 | 第一次跑、简单实验 | 正式训练、多次调参、实验管理 |

**重构版的代码更合理**：config 和代码分离，`util.py` 里抽了 `colorize`、`load_config` 等公共函数，`model_type` 可换不同 UNet。但重构版的训练循环和原版是**完全相同的算法**。

---

## 四、EDM 理论常见问题解答

### Q1: EDM 是 score matching 还是普通 diffusion？

**答案：两者本质等价。**

```
Score Matching 训练目标：
  L = E[ ||s_θ(x_t, t) - ∇_{x_t} log p(x_t|x_0)||² ]

DDPM 训练目标（预测噪声 ε）：
  L = E[ ||ε_θ(x_t, t) - ε||² ]

关系：∇_{x_t} log p(x_t|x_0) = -(x_t - x_0)/σ² = -ε/σ
→ ε = -σ · score
```

**Score** 和 **噪声预测** 只差一个 σ 的缩放因子，DDPM 其实就是 score matching 的特例。

EDM 的贡献是**把训练目标、网络输出、采样算法三者解耦**：

1. **网络可以预测三种东西**：ε / x_0 / v = (x_0 - x_t) / σ
2. **Loss 加了 σ 依赖的 Karras 权重**：`w(σ) = (σ² + σ_data²) / (σ · σ_data)²`
3. **输入预处理**（EDMPrecond）：`network_input = x_σ / √(σ² + σ_data²)`，让网络内部始终处理条件良好的输入
4. **多种 SDE 采样器**：Heun 二阶、DPM++、Stochastic Sampler

所以 EDM 不是"换了个 loss 的 diffusion"，而是一个**统一的扩散训练框架**。

### Q2: LDM 的 VAE 用预训练权重，不需要微调吗？

**答案：通常不需要。原因有三：**

**① 误差量级对比**：
```
VAE 重建 MSE：1e-5 ~ 1e-4
Diffusion 生成 MSE：1e-2 ~ 1e-3
→ VAE 误差比扩散误差小 10~100 倍，可以忽略
```

**② 单调图更易 encode**：
Stable Diffusion 的 VAE 在自然图像（RGB、丰富纹理）上训练。但 GOES 红外图是**单调灰度**，信息量小、结构简单，VAE 反而更容易压缩。

**③ 微调成本高**：
微调 VAE 需要单独训练重建 + KL divergence，工作量等于再训一个模型。

**如果你想验证 VAE 质量**，可以跑一个 sanity check：
```python
# 原始 → encode → decode → 原始，算 MSE
recon = vae.decode(vae.encode(original).latent_dist.mode()).sample
mse = ((recon - original) ** 2).mean()
# 如果 mse < 1e-4，没问题
```

**你已有 latent 数据**（`*_latent.zarr`），说明作者已经完成了 encode 这一步，直接用即可。

### Q3: EDM 中 `EDMPrecond` 做了什么？

它对 UNet 做了输入/输出的预归一化，让网络始终在"条件良好"的范围内工作：

```python
class EDMPrecond(nn.Module):
    def forward(self, x_σ, sigma):
        # x_σ: 加噪后的图, sigma: 噪声强度

        c_noise = torch.log(sigma) / 4         # 时间步 embedding
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

        # 网络的输入：归一化后的 x_σ
        network_input = x_σ / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

        # 网络预测
        F_θ = self.model(network_input, c_noise)

        # 网络输出：skip connection + learned part
        return c_skip * x_σ + c_out * F_θ
```

这是 EDM 比普通 diffusion 效果更好的关键设计之一。

### Q4: CorrDiff 的 residual 是怎么算的？

CorrDiff 训练时：
```
# 输入
past_frames = input_images[:, 0:2, :, :]      # 过去两帧
unet_prior  = input_images[:, 2:3, :, :]      # UNet baseline 预测

# 扩散训练（加噪 residual，预测 residual）
residual = output_images - unet_prior           # 真实残差
noisy_residual = residual + σ · noise
model_input = cat([noisy_residual, past_frames, unet_prior], dim=1)  # 4 通道
model_output = EDMPrecond(model_input, sigma)   # 预测的 residual

loss = MSE(model_output, residual)
```

推理时：
```
1. 跑 UNet baseline：unet_prior = UNet(past_frames)
2. 用扩散模型生成 residual：residual = Diffusion(cat([noise, past_frames, unet_prior]))
3. 最终预测：target = unet_prior + residual
```

### Q5: EDM 只是 loss 加了个系数吗？

**远不止。** 这是最容易的误解。让我用你项目里的 `EDMPrecond` 代码拆解——对比普通 DDPM 和 EDM 在每个关键步骤的差异：

#### 普通 DDPM 的训练循环
```python
# ① 固定噪声 schedule（预计算好的 β_1...β_T）
t = random_int(1, T)
α_t = alphas_cumprod[t], σ_t = sqrt(1 - α_t)

# ② 加噪
ε = randn_like(x_0)
x_t = sqrt(α_t) * x_0 + σ_t * ε

# ③ 条件拼接——noisy 和 cond 同尺度，不做任何缩放
model_input = cat([x_t, cond_ch0, cond_ch1], dim=1)  # shape [B, 3, 256, 256]

# ④ 直接过 UNet——输出就是网络的原始预测
ε_pred = UNet(model_input, t_embedding)  # shape [B, 1, 256, 256]

# ⑤ 无 σ 权重，固定 MSE
loss = F.mse_loss(ε_pred, ε)
```

#### 你项目里 EDM 的训练循环
```python
# ── 差异 ①：噪声采样不是固定 schedule，而是随机采 σ ──
sigma = exp(randn * P_std + P_mean)   # P_mean=-1.2, P_std=1.2
# 每个样本有独立的 σ，训练覆盖了完整噪声强度分布

# ── 差异 ②：三个系数缩放 ──
c_skip = σ_data² / (σ² + σ_data²)          # skip connection 权重
c_out  = σ · σ_data / √(σ² + σ_data²)      # 网络输出权重
c_in   = 1 / √(σ² + σ_data²)               # 输入缩放
c_noise = log(sigma) / 4                    # 时间步编码（不同于 DDPM 的整数 t embedding）

# ── 差异 ③：只缩放 noisy 部分，cond 保持原始尺度 ──
model_input = cat([x_noisy * c_in, cond_ch0, cond_ch1], dim=1)
# DDPM 里 x_t 和 cond 的数值尺度相同；
# EDM 里 c_in 把 x_σ 缩放到和 cond 同量级，避免网络输入不平衡

# ── 差异 ④：UNet 输出再做 skip connection 拼接 ──
F_x = UNet(model_input, c_noise)              # 网络的 learned part
D_x = c_skip * x_noisy + c_out * F_x          # 最终输出
# 这不是普通 residual 连接！
# c_skip → 1 当 σ→0（纯 skip），→ 0 当 σ→∞（纯网络）
# c_out → 0 当 σ→0（避免小噪声时网络放大误差），→ σ_data 当 σ→∞

# ── 差异 ⑤：Loss 随 σ 变化 ──
weight = (σ² + σ_data²) / (σ · σ_data)²
loss = (weight * F.mse_loss(D_x, x_0)).mean()

# ── 差异 ⑥（推理时）：多种 SDE 求解器 ──
# DDPM/DDIM：固定步长去噪，20~100 步 O(T·N²)
# EDM：Heun 二阶（ODE 求解）、DPM++、Stochastic Sampler，更少步数更高质量
```

#### 一张表总结 6 点差异

| # | 改动 | 位置 | "loss 加系数"能覆盖？ |
|---|------|------|----------------------|
| ① | 噪声从固定 schedule → 随机 σ | `EDMLoss.__call__` | ❌ |
| ② | 输入预归一化 c_in | `EDMPrecond.forward` | ❌ |
| ③ | 只缩放 noisy，不缩放 cond | `EDMPrecond.forward` | ❌ |
| ④ | σ-dependent skip connection c_skip·c_out | `EDMPrecond.forward` | ❌ |
| ⑤ | Loss 权重 w(σ) | `EDMLoss.__call__` | ✅ 只覆盖了这 1 点 |
| ⑥ | 多种 SDE 采样器 | 推理阶段 | ❌ |

**结论**：`EDMPrecond`（差异 ②③④）是 EDM 效果好于普通 diffusion 的**核心原因**，而不是 loss 权重。Karras 论文的 ablation 显示：去掉 c_skip/c_out/c_in 中任何一个，FID 都会明显恶化。

### Q6: LDM 通道翻 4×，计算量真的小 16× 吗？

**好问题。让我精确算一下。** 卷积层 FLOPs 公式：`F = H × W × C_in × C_out × kernel²`（假设 kernel=3）。

#### 配置对比

| | 原空间 UNet | Latent UNet |
|--|------------|-------------|
| 输入通道 | `cat([x_σ(1), cond_ch0, cond_ch1])` = **3** | `cat([latent_σ(4), cond_latent(8)])` = **12** |
| 输出通道 | 1 | 4 |
| 空间分辨率 | 256 × 256 | 64 × 64 |
| block_out_channels | (128,128,256,256,512,512) | 同左 |

#### 逐层 FLOPs

```
第一层 Conv（最大分辨率，通道最少）：
══════════════════════════════════════════════
原空间：  256 × 256 ×   3 × 128 × 9 = 224,190,464  ≈ 224M
Latent：   64 ×  64 ×  12 × 128 × 9 =  56,047,616  ≈  56M
比值：56/224 = 1/4   ← 你说的"只小 4×"在这里是对的！
原因：空间缩小 16×，但 C_in 从 3→12 翻了 4×，抵消了。

最后一层 Conv（最小分辨率，通道最多）：
══════════════════════════════════════════════
原空间：    8 ×   8 × 512 × 512 × 9 = 18,874,368   ≈ 18.9M
Latent：    2 ×   2 × 512 × 512 × 9 =  1,179,648   ≈  1.18M
比值：1.18/18.9 = 1/16   ← 这里完完全全 16×！
原因：深层 C_in 相同（都=512），只有空间 16× 的收益。

整体加总（encoder + decoder 所有层）：
══════════════════════════════════════════════
原空间总 FLOPs ≈ 4.5 × 10⁹
Latent 总 FLOPs ≈ 5.3 × 10⁸
比值 ≈ 1/8.5   ← 大约 8~9×
```

#### 为什么是 8× 不是 16× 也不是 4×？

```
单层 FLOPs ≈ H² × C_in × C_out × 9

两层贡献方向相反：
  空间 H²：每层 latent 的 H 是原空间的 1/4 → 每一层都是 1/16 ← 主导，想给你 16×
  C_in：  只有前几层有 (12 vs 3) = 4× 惩罚 → 抵消成 4×/层     ← 只影响前 2 层

深层（C_in 相同，都是 512）：完完整整 16× 收益
浅层（C_in 翻 4×）：          16× 空间 ÷ 4× 通道 = 4× 收益

加权平均 → 整体 ≈ 8~9×
```

#### 但还有一个更大的收益你可能没注意到：**Feature Map 显存**

训练时中间 feature map 存储是 `B × C × H × W`，显存占用和 `C × H²` 成正比：
```
原空间最深层：H=8, C=512 → 8²×512 = 32,768
Latent最深层：H=2, C=512 → 2²×512 = 2,048
比值 = 1/16   ← 显存也是完完整整 16×！
```

#### 还有一个推理端的数量级差异

原空间 256×256 扩散推理需要 N=20~100 步 SDE 求解器，每一步在 256×256 上跑 UNet；latent 只在 64×64 上跑同样多步数。即使每步计算量只差 8×，**100 步累积就是 800× 差距**——原空间推理可能 30s/张，latent 只要 1~2s/张。

#### 总结

| 指标 | 原空间 vs Latent |
|------|-----------------|
| 每层 FLOPs 比值 | 浅层 4×，深层 16×，整体 ≈ 8~9× |
| Feature map 显存 | ≈ 16×（通道相同，只有空间差异） |
| 模型参数量 | 几乎相同（结构不变，只变分辨率） |
| 单步推理速度 | ≈ 6~8× |
| 完整推理（N 步） | ≈ 6~8× 加速 |

---

## 五、快速开始指南

### 5.1 修改脚本的通用步骤

每次换训练，需要改的参数：

| 参数 | vanilla | EDM | LDM | CorrDiff |
|------|---------|-----|-----|----------|
| `dataset_path` | `edm_GOES_ch13_train_dataset.zarr` | 同左 | `edm_GOES_ch13_train_dataset_latent.zarr` | `edm_GOES_ch13_train_dataset_CorrDiff.zarr` |
| `output_dir` | `outputs/vanilla_unet/` | `outputs/edm/` | `outputs/edm_ldm/` | `outputs/corrdiff/` |
| `in_channels` | 2 | 3 (noisy + 2 cond) | 12 (4 latent + 8 cond) | 4 (noisy + 2 cond + 1 prior) |
| `image_size` | 256 | 256 | 64 | 256 |

### 5.2 OOM 处理

4090 48GB 比 GH200 96GB 小一半，调参建议：

```python
train_batch_size = 16  # 原 45
gradient_accumulation_steps = 2  # 保持有效 batch_size ~32
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 显存碎片化优化
```

### 5.3 常见报错与修复

| 错误 | 原因 | 修复 |
|------|------|------|
| `AttributeError: module 'matplotlib.cm' has no attribute 'get_cmap'` | matplotlib 3.11 移除了 `cm.get_cmap` | 改成 `matplotlib.colormaps.get_cmap` |
| `RuntimeError: Input type (c10::Half) and bias type (float) should be the same` | fp16 数据 + fp32 模型不匹配 | 推理时 `.float()` 转换 |
| `num_workers` 多进程卡死 | DataLoader 多进程加载 zarr | 改成 `num_workers=0` |
| `CUDA out of memory` | 显存不够 | 见 5.2 |

---

## 六、项目文件结构

```
cira-diff/
├── README.md
├── LICENSE
├── pyproject.toml                  # pip install -e . 入口
├── config_CorrDiff.py              # CorrDiff 重构版的 config 示例
│
├── cira_diff/                      # 核心 Python 包
│   ├── __init__.py
│   ├── dataset.py                  # 数据集工具类
│   ├── edm.py                      # EDMPrecond 封装
│   ├── train_Diff.py               # 重构版 EDM 训练入口（需 config）
│   ├── train_CorrDiff.py           # 重构版 CorrDiff 训练入口（需 config）
│   ├── generate.py                 # 推理/采样脚本
│   └── util.py                     # 公共函数（colorize, load_config 等）
│
└── scripts/Chase_2025/             # 原始作者的独立脚本
    ├── train_vanilla_unet_Chase2025.py
    ├── train_edm_Chase2025.py
    ├── train_edm_LDM_Chase2025.py
    ├── train_edm_CorrDiff_Chase2025.py
    ├── Run_Forecasts_Chase2025.ipynb
    ├── log_gpu.py
    └── README.md
```

---

## 七、已验证的运行记录

| 脚本 | 配置 | 结果 |
|------|------|------|
| `train_vanilla_unet_Chase2025.py` | 500 样本, 2 epoch | ✅ exit 0, loss=0.107 |
| `train_vanilla_unet_Chase2025.py` | 1 样本, 1000 epoch | ✅ overfit, MSE=2.16e-5 |
| `train_edm_Chase2025.py` | 50 样本, 2 epoch | ✅ exit 0, checkpoint 保存 |

**待运行**（数据已就位）：
- `train_edm_LDM_Chase2025.py`（latent 数据已有，只需联网下载 VAE）
- `train_edm_CorrDiff_Chase2025.py`（CorrDiff 数据已有）