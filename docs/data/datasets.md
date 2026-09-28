# 数据集清单

| 数据集 | 角色 | 当前状态 | 已验证事实 |
| --- | --- | --- | --- |
| CIRA-Diff GOES derived Zarr | 当前 baseline/reproduction | `AUDITED-DOCUMENTARY`，runtime pending | GOES-16 ABI Channel 13、10 min、256×256；shape 见 `cira.md` |
| CIRA-Diff latent Zarr | 当前 baseline latent path | `AUDITED-DOCUMENTARY`，runtime pending | shape 已在 `cira.md` 记录；source lineage 仍不完整 |
| Himawari-8/9 | 未来卫星研究线 | `RESEARCH LEAD`，未验证 | 仅有任务提示级上下文；见 `himawari.md` |
| NOAA GOES Open Data raw archive | CIRA-Diff 数据的 upstream source | `REFERENCED`，本地不存在 | 既有 docs 记录 source 为 NOAA GOES Open Data |

## 可用性边界

[FACT] 当前 Git tree 没有 tracked data store。既有项目 docs 记录了远端访问和此前声称的本地 subset；这些 payload 不属于本次变更。

## 未来 dataset manifest 必须包含

Dataset ID、source URL/path、sensor/platform、channel、calibration/unit、projection/grid、spatial resolution、temporal interval、region、sample index 或 timestamp、train/validation/test membership、normalization statistics、checksum 和 license/access constraints。
