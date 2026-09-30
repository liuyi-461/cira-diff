# SimVP ↔ OpenSTL Provenance 审计

> 目的：澄清 cira-diff reproduction `README.md` 中标记的
> `Official repository: TODO — verify；需要审计与 OpenSTL 的关系`。
> 审计日期：2026-09-29。状态语义见 `docs/reproduction/README.md`。

## 结论（核心）
**OpenSTL 就是 SimVP 的官方实现宿主（official repository）。** SimVP 并非一个独立的
第三方仓库，而是由同一作者团队（CAIRI / Westlake University：Gao, Tan, Li, Wu）在
OpenSTL 基准中以 `method='SimVP'` 的形式提供的。因此本 reproduction 直接 clone
OpenSTL 即等价于获取 SimVP 的官方参考实现，二者是「方法 ⊆ 基准」的包含关系。

## 上游来源
- 仓库：`https://github.com/chengtan9907/OpenSTL`（setup.py 中 `url` 字段一致）。
- 归属：CAIRI AI Lab, Westlake University（setup.py `author='CAIRI Westlake University Contributors'`）。
- 文档：https://openstl.readthedocs.io ，页脚标注 © CAIRI，revision `41831fad`
  （可作为文档快照参考，不等于代码 commit）。

## 论文与代码对应关系
| 方法 | 论文 | 在 OpenSTL 中的实现 |
| --- | --- | --- |
| SimVP v1 | CVPR'2022, Gao et al., arXiv:2206.05099 | `model_type='IncepU'` → `MidIncepNet` |
| SimVP v2 | arXiv:2211.12509, Tan et al. | `model_type='gSTA'`（默认）→ `MidMetaNet` + `GASubBlock` |
| TAU | CVPR'2023, Tan et al., arXiv:2206.12126 | `model_type='tau'` → `TAUSubBlock` |
| OpenSTL（基准）| NeurIPS'2023 D&B, Tan, Li et al., arXiv:2306.11249 | 三层抽象 `api/methods/models/modules` |

## Upstream commit（待 pin）
- 本地 clone 在本 workspace 中没有独立 git commit（cira-diff 仓库整体 `No commits yet`）。
- **待办**：从 `github.com/chengtan9907/OpenSTL` 拉取并 `git rev-parse HEAD` 记录 revision，
  回填 reproduction `README.md` 的 `Upstream commit` 字段，并固定 `environment.yml` 锁版本。

## 审计发现的偏差 / 风险
1. **API 漂移**：本仓库的 `SimVP_Model` 构造签名为
   `(in_shape, hid_S, hid_T, N_S, N_T, model_type, ...)`，
   而 `tests/test_models/test_simvp.py` 仍使用旧的 `SimVP_Model(arch=...)` /
   `SimVP_Model(num_layers=3, num_hidden=1)` 并假设 `init_weights()` 存在——
   均**已不存在**，导致该测试必然失败。已修复（见 `tests/test_models/test_simvp.py`）。
2. **无独立 SimVP 仓库**：reproduction `README.md` 中 `Official repository` 应改为指向
   OpenSTL（或其 SimVP 子文档），而非寻找单独 repo。
3. **License**：Apache-2.0，可合法用于 reproduction 与对比实验。

## 状态迁移记录
- `Surveyed`：SimVP 方法与 OpenSTL 关系已审计（本文）。
- 下一步：`Code Available`（本 clone 即代码）→ `Downloaded` → `Environment Working`。
- 迁移须附 command / revision / artifact / deviation（见 `docs/reproduction/README.md`）。
