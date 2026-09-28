# Reproduction 登记

Reproduction 记录保存从外部论文/仓库到本地 verified understanding 的完整路径。它不是 model zoo，也不表示已经下载代码或 weights。

## 状态词汇

`Surveyed` → `Code Available` → `Downloaded` → `Environment Working` → `Inference Reproduced` → `Training Reproduced` → `Evaluation Reproduced`。

每次状态迁移都必须有 command、revision、artifact 和 deviation note。只有论文或 repository 链接不构成 reproduction result。

## 登记表

| 条目 | 角色 | 当前状态 |
| --- | --- | --- |
| [CIRA-Diff](cira_diff/) | 当前 satellite baseline | source available；data audit documented；runtime pending |
| [PredRNN++](predrnnpp/) | recurrent/video-prediction comparison | surveyed/planned；未下载 |
| [ConvLSTM](convlstm/) | classic deterministic control | candidate only；未下载 |
| [SimVP](simvp/) | low-cost deterministic comparison | candidate only；未下载 |
| [TAU](tau/) | temporal modeling comparison | candidate only；未下载 |
| [Earthformer](earthformer/) | transformer/Earth-system comparison | candidate only；未下载 |
| [DaYu](dayu/) | satellite-specific research lead | literature/repository details unresolved |
| [FY-4A DDMS](fy4a_ddms/) | satellite-specific research lead | literature/repository details unresolved |
| [Himawari PredRNN++](himawari_predrnnpp/) | satellite-specific research lead | literature/data details unresolved |

在记录中出现 verified status、command 和 artifact 前，不得把任何条目描述为 locally reproduced。
