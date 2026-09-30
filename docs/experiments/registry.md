| ID | 记录 | 问题 | 状态 | 证据 |
| --- | --- | --- | --- | --- |
| EXP-009 | [Single-sample Overfit Gate](gate_checks/EXP-009-single-sample-overfit.md) | Vanilla UNet / EDM / LDM / CorrDiff 单样本过拟合 | ✅ Completed | `outputs/overfit_comparison/` |
| EXP-010 | [Vanilla UNet Full Training](baselines/EXP-010-vanilla-unet-full-training.md) | 复现 UNet baseline 的完整训练 | ✅ Completed (epoch 1000/1000) | `outputs/vanilla_unet_full/` |
| EXP-011 | [EDM Full Training](baselines/EXP-011-edm-full-training.md) | 复现 EDM diffusion baseline 完整训练 | 🔄 **Resuming** (Job 109 GPU 2, epoch 330→1000, batch 24→20 防 OOM, val_loss NaN bug fixed, torch.cuda.empty_cache added) | `outputs/edm_full/` |
| EXP-012 | [LDM Full Training]() | 复现 LDM baseline | 📋 Planned | - |
| EXP-013 | [CorrDiff Full Training](baselines/EXP-013-corrdiff-full-training.md) | 复现 CorrDiff correction diffusion baseline | 🔄 **Training** (Job 107 GPU 0, epoch 15/1000, batch 22×4=88 eff, val_loss NaN→sigma=0.002 workaround) | `outputs/corrdiff_full/` |
| EXP-014 | [Vanilla UNet Seed 0](baselines/EXP-014-vanilla-unet-seed0-training.md) | seed=0 复现 UNet baseline | ✅ Completed | `outputs/vanilla_unet_seed0/` |
| EXP-015 | [Vanilla UNet Seed 123](baselines/EXP-015-vanilla-unet-seed123-training.md) | seed=123 复现 UNet baseline | ✅ Completed | `outputs/vanilla_unet_seed123/` |

**评估类**:
| EXP-EVAL-001 | [UNet Trained vs OpenSource](evaluation/EXP-EVAL-001-vanilla-unet-baseline-vs-opensource.md) | UNet seed=0 vs opensource checkpoint | ✅ Completed |
| EXP-EVAL-002 | [EDM Trained vs OpenSource](evaluation/EXP-EVAL-002-edm-baseline-vs-opensource.md) | EDM @ 287ep vs opensource @ 1000ep | ✅ Completed |
| EXP-EVAL-003 | [Multi-seed UNet Rollout](evaluation/EXP-EVAL-003-multi-seed-vanilla-unet-rollout.md) | 3 seed × 365 day rollout | ✅ Completed |
| EXP-EVAL-004 | [CorrDiff Trained vs OpenSource](evaluation/EXP-EVAL-004-corrdiff-baseline-vs-opensource.md) | 本地 CorrDiff vs opensource checkpoint | 📋 Pending |

**当前 Slurm 作业**:
- GPU 0: Job 107 **CorrDiff Full Training** (epoch 15/1000, 45.7 GB, ~1.4 it/s, ETA ~13d)
- GPU 1: Job 108 drdd_sevir (group2)
- GPU 2: Job 109 **EDM Full Training Resume** (epoch 330/1000, 41.7 GB, ~1.5 it/s, ETA ~9.5d)
- GPU 3-7: 空闲

**关键 Bug Fixes (2026-09-30)**:
1. **EDM OOM Cumulative**: batch_size 24→20, epoch 末加 `torch.cuda.empty_cache()`, GPU 从 43.5 GB→41.7 GB (6.9 GB free)
2. **EDM/CorrDiff val_loss NaN**: `sigma=0` → `EDMPrecond` 中 `log(0)=-inf` → NaN。修复为 `sigma=0.002`（min noise level）。注意：此 sigma 远低于 EDM 训练范围（[0.09, 1.0]），val loss ≈ 0 无科学意义。
3. **CorrDiff 硬伤**: hardcoded paths → 相对路径；全 CPU 加载 → lazy loading；无验证 → 独立验证 zarr + best checkpoint。
