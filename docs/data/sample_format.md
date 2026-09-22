# Sample Format

## 当前 CIRA-Diff 接口

[FACT] `cira_diff.dataset.ZarrDataset` 返回：

```python
target, condition = dataset[index]
```

该顺序必须明确记录，因为它不同于许多 `(condition, target)` API。

| 路径 | Tensor 含义 | Shape |
| --- | --- | --- |
| ordinary train target | 下一时刻 normalized IR frame | `(1, H, W)` |
| ordinary train condition | 两帧历史 normalized IR frame | `(2, H, W)` |
| CorrDiff train condition | 两帧 history + first guess | `(3, H, W)` |
| validation/test target | 18 帧 future truth | `(18, H, W)` |
| validation/test condition | 两帧历史帧 | `(2, H, W)` |

## 未来 common contract（计划）

```text
history:  [B, H, C, Y, X]
target:   [B, F, C, Y, X]
forecast: [B, F, C, Y, X]
```

[PLAN] CIRA-Diff 行为测试完成后，可增加 common adapter。它必须在 reproduction boundary 保留 legacy target/condition order，并记录任何 transpose 或 inverse-normalization。
