# SimVP Reproduction 记录

| 字段 | 记录 |
| --- | --- |
| 角色 | low-cost deterministic multi-frame comparison |
| Paper | `TODO — verify primary source`（待 pin DOI/arXiv，见 `OpenSTL/docs/reproduction/provenance_audit.md`） |
| Official repository | `github.com/chengtan9907/OpenSTL` —— 已审计：OpenSTL **就是** SimVP 的官方宿主（同作者团队），SimVP v1 `IncepU` / v2 `gSTA` / TAU 均以 method 形式内置，无独立 SimVP 仓库 |
| Upstream commit | `TODO` —— 本地 clone 未携带独立 git 历史，需从上游 pin 后回填 |
| Dataset | EDM GOES-16 ABI channel-13 zarr；256×256 float16；2 帧 → 1 帧（`OpenSTL/docs/experiments/cira_diff/EXP-001-cira-diff-data-audit.md`） |
| Original task/input/output | `x=(2,1,256,256)` → `y=(1,1,256,256)`，10 min 间隔 |
| Environment | **Working**（conda env `openstl`：torch 2.6.0+cu124 / lightning 2.6.6 / timm 0.9.12 / fvcore 0.1.5；配方见 `OpenSTL/docs/dev/environment.md`） |
| Weights | `TODO`；未下载（无官方 GOES-13 权重，走自训练） |
| Reproduction status | `Surveyed → Code Available → Environment Working → Training Reproduced (smoke, EXP-002 on GPU)`；EXP-003 仅 CPU 验证、**未**在 GPU 上正式跑 |
| Local modification | 无 —— 上游 `openstl/` 未改动任何一行，全部为新增文件（adapter / config / tools / docs） |
| Data adapter | 已完成：`openstl/datasets/dataloader_goes13.py` |
| Evaluation adapter | 复用 OpenSTL 原生（`mse` / `mae` / `rmse`，`saved/metrics.npy`） |
| Known issues | ① direct multi-step output 必须与 autoregressive CIRA-Diff 在同等多步协议下公平比较；② `timm` 必须 <1.0；③ 全量 split 用 `preload=True` 约需 28 GB 内存，须改 `--no-preload` |

## 状态迁移日志

- **2026-09-30 — Environment Working**
  - command：见 `OpenSTL/docs/dev/environment.md`（`conda create -n openstl --clone test` + `pip install timm==0.9.12 opencv-python-headless imageio imageio-ffmpeg PyWavelets scikit-image`）
  - artifact：极小程序跑（`--n-train 8 --n-val 4 --n-test 4 --epochs 1 --cpu`）产出
    `checkpoints/{best,last,best-epoch=00-val_loss=0.348}.ckpt`、
    `saved/{inputs,preds,trues,metrics}.npy`、`model_param.json`
  - deviation：新建专用 env 而非改动共享的 `test`（timm 必须降级），见 `DBG-005`
- **2026-09-30 — 参考脚本删除**
  - `OpenSTL/test_dl/test_single_sample_Chase2025.py` 已从工作区删除；仅 4 处文档/注释引用失效，
    不影响运行，详见 `DBG-006`
