# Debug 记录（Dev）

## DBG-001 过期单元测试
- 现象：`tests/test_models/test_simvp.py` 中 `SimVP_Model(arch='unknown')`、
  `SimVP_Model(num_layers=3, num_hidden=1)` 与当前构造签名
  `(in_shape, hid_S, hid_T, N_S, N_T, model_type, ...)` 不匹配，断言触发非预期。
- 原因：测试未随模型接口更新。
- 解法（建议）：重写测试以匹配现有签名（用 `in_shape` 构造并跑一次 forward），
  不修改模型接口。
- 状态：未修复（candidate only）。

## DBG-002 分辨率下采样约束
- 现象：输入 H/W 需满足 `int(H/2**(N_S/2))` 为整数。
- 解法：选定数据集时确认空间分辨率能被 2^(N_S/2) 整除，或调整 `N_S`。
