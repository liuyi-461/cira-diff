# 术语表

## STL
全称：Spatio-Temporal predictive Learning（时空预测学习）。
说明：对时序空间场（视频、天气、交通）进行未来帧预测的任务族。

## SimVP
全称：Simpler yet Better Video Prediction（CVPR'2022）。
说明：纯 CNN 编码-隐空间翻译-解码的确定性视频预测模型。

## MetaFormer
说明：以 "token mixer + MLP" 为统一范式的架构族；SimVP v2 用其作为隐空间翻译骨干。

## gSTA
全称：Global Spatial-Temporal Attention（SimVP v2 默认骨干）。
说明：大核注意力（LKA）空间门控单元 + MLP，归一化用 BatchNorm2d。

## TAU
全称：Temporal Attention Unit（CVPR'2023）。
说明：在 gSTA 基础上追加 SE 通道注意力的时序建模单元（`TAUSubBlock`）。

## Direct multi-step
说明：模型一次性输出全部未来帧；SimVP 默认行为。

## Autoregressive
说明：将上一步预测作为下一步输入递推生成；CIRA-Diff 采用，SimVP 在 `aft>pre` 时也可切换。

## pre_seq_length / aft_seq_length
说明：输入历史帧数 / 输出未来帧数。
