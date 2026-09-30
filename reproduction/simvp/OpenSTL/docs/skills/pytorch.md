# 深度学习开发规范（Skill）

## 框架
- OpenSTL 基于 PyTorch + PyTorch Lightning（`openstl/api/exp.py` 的 `BaseExperiment`）。

## 约定
- 训练：`tools/train.py -d <dataname> --lr ... -c configs/.../SimVP_gSTA.py --ex_name ...`
- config 优先级：`--overwrite` 决定是否用 config 覆盖 CLI 默认（见 `tools/train.py`）。
- 不要直接改 `openstl/` 核心；通过 config 与 adapter 扩展。

## 避坑
- `tests/test_models/test_simvp.py` 已过期，运行前需先修复（见 `docs/dev/debugging.md`）。
- 输入分辨率需满足 `N_S/2` 次 2× 下采样（`int(H/2**(N_S/2))`）。
