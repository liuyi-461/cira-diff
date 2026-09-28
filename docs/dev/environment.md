# 开发环境

## 本地仓库

- Repository: `/Users/liuyi/Projects/cira-diff`
- Python requirement: `>=3.8`（`pyproject.toml`）
- Package: `cira_diff` version `0.0.2`
- Upstream: `https://github.com/dopplerchase/cira-diff`

## 上游 README 提到的依赖

PyTorch、torchvision、torchaudio、Diffusers、Transformers、Accelerate、matplotlib、TensorBoard、`py3nvml`、build、Zarr。实际 CUDA/PyTorch 组合必须按目标 GPU 单独确认。

## 开发原则

- 先运行静态检查和 Dataset smoke test，再申请 GPU 训练。
- 所有机器相关路径放入 config，不直接改成个人绝对路径后提交。
- 训练、推理、评估都记录命令、Git commit、数据版本和输出目录。
- 不在本地工作树提交真实数据 payload、checkpoint 或 TensorBoard 大文件。

## 当前环境状态

[UNKNOWN] 本轮未创建或修改 Python environment，也未运行 GPU 训练；不能声称依赖已安装或 baseline 已通过 smoke test。
