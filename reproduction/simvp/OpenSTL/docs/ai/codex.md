# Codex 使用规范

## 适合
- Repo 级开发（批量生成 adapter、修复测试）。
- 自动测试（补全 / 修正 `tests/`）。

## 使用注意
- 修复 `tests/test_models/test_simvp.py` 时，使其匹配当前 `SimVP_Model` 构造签名，
  而非修改模型接口。
- 运行后需在 `docs/dev/debugging.md` 记录问题与解法。
