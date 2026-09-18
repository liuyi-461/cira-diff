# Glossary

## CIRA-Diff

CIRA 开源的条件 score-based diffusion 研究代码；本项目的第一阶段 baseline。

## GOES / ABI

Geostationary Operational Environmental Satellite / Advanced Baseline Imager。官方 baseline 使用 GOES-16 ABI Channel 13。

## Brightness Temperature（BT）

卫星红外观测常用的等效辐射温度。CIRA-Diff 预测的是 10.3 μm 红外亮温图像。

## Condition

模型输入的历史图像。在官方任务中是 t−10 min 与 t 两帧。

## Target / Generation image

模型需要生成的下一时刻图像，在训练中记为 clean image；官方单步目标为 t+10 min。

## EDM

Elucidating the Design space of Diffusion-Based Generative Models。CIRA-Diff 使用其预条件、噪声采样和采样思想。

## Diff

直接对下一时刻图像进行条件扩散预测的变体。

## CorrDiff

先由普通 U-Net 预测，再对 `truth - U-Net prediction` 的 residual 做扩散建模的变体。

## LDM

Latent Diffusion Model。先用 VAE 将图像压缩到 latent space，再在 latent 上进行扩散。

## Autoregressive rollout

将上一步预测结果反馈为下一步条件，重复单步模型以获得多步/长时效预测。

## Satellite-only forecasting

只使用卫星观测序列作为输入，不引入 NWP、雷达或其他大气状态变量。

## Exposure bias

训练阶段主要看真值历史、推理阶段却看模型历史，导致 rollout 中输入分布发生偏移的问题。
