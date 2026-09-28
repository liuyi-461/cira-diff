# 科学问题

状态词汇：`OPEN`、`UNTESTED`、`PARTIALLY INFORMED`、`ANSWERED`。

## RQ-001：哪种时间建模范式适合卫星云图预报？

**状态：** OPEN / UNTESTED

比较：

```text
范式 A：(X[t-1], X[t]) → X̂[t+1] → autoregressive rollout
范式 B：(X[t-n+1], …, X[t]) → (X̂[t+1], …, X̂[t+m])
```

[EVIDENCE] CIRA-Diff 是范式 A 的具体案例：训练时 `2 → 1`，推理时进行 18-step autoregressive inference。

[UNKNOWN] 本项目没有受控证据证明任一范式普遍优于另一范式。

## RQ-002：增加历史长度带来什么收益？

**状态：** OPEN / UNTESTED

比较 2、4、8、12、24 帧 history 对短 lead time、长 lead time、云场运动、生成、增强、消散和组织结构的影响。必须控制参数量、计算量和数据量变化。

## RQ-003：短历史能否区分平流与云场内禀演变？

**状态：** OPEN / UNTESTED

两帧可能提供一阶位移线索，但云生成、发展、消散和非刚性形变可能需要更长时间上下文或其他预测变量。这是研究问题，不是预设结论。

## RQ-004：autoregressive rollout 误差如何累积？

**状态：** OPEN / PARTIALLY INFORMED

必须分别测量 single-step skill 和 lead-time degradation，并区分单步误差、exposure/distribution shift、重复随机采样和天气型变化的作用。单步分数好不等于 3 小时 rollout 好。

## RQ-005：diffusion 增加的是 realism、不确定性，还是 forecast skill？

**状态：** OPEN / PARTIALLY INFORMED

分开评价：

- deterministic pixel/structural skill；
- ensemble sharpness 与 calibration；
- perceptual/visual plausibility；
- cloud-object 和 meteorological event skill。

[EVIDENCE] 既有 literature record 说 CIRA-Diff 论文报告了 ensemble 与 cold-cloud 分析。[UNKNOWN] 本仓库尚未复现这些结果。

## RQ-006：可预测性如何依赖空间尺度与结构？

**状态：** OPEN / UNTESTED

比较大尺度云场组织/位移与小尺度纹理、云边界、对流核心。不能把一个全像素聚合分数当成完整任务评价。

## RQ-007：卫星预报 skill 是否依赖天气型？

**状态：** OPEN / UNTESTED

在 labels 和数据谱系充分时，分别研究普通云场、冷云/深对流候选区域以及热带气旋云系组织。没有验证的事件定义和 case 选择协议时，不得声称 event skill。

## RQ-008：结论能否跨传感器和数据集迁移？

**状态：** OPEN / UNKNOWN

只有在 calibration、projection、spatial resolution、temporal sampling、region 和 split policy 明确后，才能比较 GOES 与 Himawari。任务提示中的 Himawari 背景是研究线索，不是已验证数据事实。

## 范围边界

卫星云图预报与以下任务相关，但不能混为同一问题：

| 任务 | 主要目标 | 为什么不是同一问题 |
| --- | --- | --- |
| Generic video prediction | 视觉帧 | 像素语义和观测几何可能未定义为物理量 |
| Precipitation nowcasting | 雷达/降水场 | 目标、threshold 和 verification 聚焦降水或水凝物发生 |
| NWP / AI weather forecasting | 大气状态或天气变量 | 通常使用多变量状态，forecast object 不同 |
| Satellite cloud forecasting | 卫星图像或亮温场 | 必须考虑 cloud-top temperature、viewing geometry、advection 和云生命周期 |

方法可以相互借鉴，但结论只有在 input、target、sampling、horizon 和 verification contract 匹配时才能迁移。
