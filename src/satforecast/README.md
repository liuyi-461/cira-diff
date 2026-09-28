# satforecast 源码包

该 package 当前只是 scaffold。只有 verified reusable implementation 才能 promotion 到这里；在 `docs/project/migration.md` 的 gate 通过前，不得将 `cira_diff/` 直接复制进来。

| Module | 目标职责 | 当前状态 |
| --- | --- | --- |
| `data/` | canonical samples、contracts 和 adapters | scaffold |
| `preprocessing/` | calibrated satellite-to-sample transforms | scaffold |
| `models/` | stable model interfaces/promoted implementations | scaffold |
| `training/` | reusable training orchestration | scaffold |
| `inference/` | single-step 和 rollout interfaces | scaffold |
| `evaluation/` | shared metric/artifact implementation | scaffold |
| `utils/` | tested generic utilities | scaffold |
