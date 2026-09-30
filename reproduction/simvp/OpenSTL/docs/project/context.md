# SimVP Reproduction（OpenSTL）项目背景

## 项目简介

说明：
- 项目目标：在 cira-diff 实验框架下，将 SimVP 作为一个 **low-cost deterministic multi-frame comparison**
  方法建立起来，用于与自回归的 CIRA-Diff 做公平对比。
- 应用场景：卫星云图 / 时空预测（spatio-temporal predictive learning, STL）基准对比。
- 解决的问题：在扩散类生成模型之外，提供一个低成本、确定性的多帧预测对照基线。

## 技术背景

说明：
- 领域背景：时空预测涵盖合成运动目标、视频、交通流、天气/卫星预测等任务。
- 核心技术：SimVP（纯 CNN 编码-隐空间翻译-解码结构），以及 OpenSTL 提供的 13 种
  MetaFormer 骨干（gSTA / ConvMixer / ConvNeXt / HorNet / MLPMixer / MogaNet / PoolFormer /
  Swin / Uniformer / VAN / ViT / TAU 等）。

## 输入数据

说明：
- 数据来源：待定（reproduction `TODO`）。OpenSTL 支持 mmnist、weather、kth、taxibj、sevir 等。
- 数据格式：`(B, T, C, H, W)`，T 为 pre_seq_length（历史帧数）。
- 数据规模：取决于所选数据集与切分（见 `docs/data/`）。

## 输出结果

说明：
- 系统输出：给定历史帧，直接输出 `aft_seq_length` 帧预测（direct multi-step）。
- 模型输出：`(B, T_out, C, H, W)` 的预测张量。

## 技术栈

- Python（<=3.10.8）
- PyTorch（Lightning 封装见 `openstl/api/exp.py`）
- timm（提供 ConvNeXt / Swin / ViT / MLPMixer 等骨干）
- OpenSTL 框架（method / model / module 三层抽象）

## 项目约束

说明：
- 不可改变的接口：OpenSTL 的 `Base_method`、`BaseExperiment` 训练/测试 API。
- 数据限制：SimVP 默认直接多帧输出，与自回归 CIRA-Diff 对比时须保证多步预测协议一致
  （见 `docs/project/decisions.md` DEC 相关条目）。
- 工程限制：reproduction 仍处 candidate only，禁止声称已 locally reproduced。
