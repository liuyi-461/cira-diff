# 常见问题（Ops）

## 训练不收敛
- 检查 mean / std 归一化是否一致。
- 检查 `pre_seq_length` / `aft_seq_length` 与 config 匹配。

## 分辨率报错
- 确认 H/W 能被 2^(N_S/2) 整除（见 `docs/dev/debugging.md` DBG-002）。

## 测试断言失败
- 优先检查是否因过期测试（`tests/test_models/test_simvp.py`，DBG-001）引起。
