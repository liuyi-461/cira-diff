# 数据系统

数据层必须在模型比较前定义 sample 的含义。

## Canonical 数据流

```text
原始卫星数据
  → calibration / physical conversion
  → spatial crop 或 patch
  → temporal sampling
  → normalization
  → history sequence
  → target sequence
  → model
  → forecast
  → evaluation
```

## 文档

- [datasets.md](datasets.md)：数据集清单和验证状态。
- [cira.md](cira.md)：CIRA-Diff 派生 GOES 数据合同。
- [himawari.md](himawari.md)：Himawari 研究线索，未解决字段保留 unknown。
- [sample_format.md](sample_format.md)：tensor layout 与时间语义。
- [preprocessing.md](preprocessing.md)：预处理合同与风险。
- [splits.md](splits.md)：split policy 与 leakage 控制。
- [data_lineage.md](data_lineage.md)：从 source 到 forecast artifact 的 provenance。

数据声明使用 `[FACT]`、`[EVIDENCE]`、`[UNKNOWN]`、`[CONFLICT]` 或 `[PLAN]`。本地 development subset 不得当作完整数据集。
