# Debug 记录（Dev）

## DBG-001 过期单元测试
- 现象：`tests/test_models/test_simvp.py` 中 `SimVP_Model(arch='unknown')`、
  `SimVP_Model(num_layers=3, num_hidden=1)` 与当前构造签名
  `(in_shape, hid_S, hid_T, N_S, N_T, model_type, ...)` 不匹配，断言触发非预期。
- 原因：测试未随模型接口更新。
- 解法（建议）：重写测试以匹配现有签名（用 `in_shape` 构造并跑一次 forward），
  不修改模型接口。
- 状态：未修复（candidate only）。

## DBG-002 分辨率下采样约束
- 现象：输入 H/W 需满足 `int(H/2**(N_S/2))` 为整数。
- 解法：选定数据集时确认空间分辨率能被 2^(N_S/2) 整除，或调整 `N_S`。

## DBG-003 argparse 选项名冲突与 dest 静默覆盖
- 现象 A（致命）：两个训练脚本原先自行注册了 `--lr`，而 `create_parser()`
  已注册同名选项 → 构造 parser 时抛
  `argparse.ArgumentError: conflicting option string: --lr`，脚本一启动就崩。
- 现象 B（静默）：新增选项与上游选项共享 dest 时（`--model-type` vs `--model_type`、
  `--batch-size` vs `--batch_size`、`--num-workers` vs `--num_workers`），argparse 只保留
  **先注册**的那个默认值，导致自设默认值失效：`model_type=None`（`MetaBlock` 抛
  `NotImplementedError`）、`batch_size=16`、`num_workers=4`。
- 解法：删掉重复的 `--lr` 注册，改在 `create_parser()` 之后调用
  `parser.set_defaults(lr=..., model_type=..., batch_size=..., num_workers=...)`。
  `set_defaults` 会覆盖**所有**共享该 dest 的 action 的默认值，两种拼写法仍都可用。
- 校验：已用正则扫描两个脚本与 `parser.py` 全部 `add_argument`，确认无同名冲突；
  `python -m py_compile` 通过。
- 状态：已修复（2026-09-30）。

## DBG-004 相对路径在 slurm 下漂移 / openstl 不可导入
- 现象：`--res_dir` 默认 `'work_dirs'`、`--output-dir` 默认 `./single_sample_goes13_output`
  均为相对路径，落点由作业工作目录（= `sbatch` 提交位置，或 `-D` 指定）决定，产物会散落。
- 现象：`python /abs/path/tools/xxx.py` 时 Python 只把**脚本所在目录**
  （`…/OpenSTL/tools`）放进 `sys.path`，而非仓库根 → `import openstl` 直接失败。
- 现象：作业超时被杀后无自动续跑（`ckpt_path` 默认 None），重投从第 0 轮开始。
- 解法：`test_dl/test_dl.slurm` 已改为
  ① 全部绝对路径（`--res_dir` / `--ex_name` / `--zarr` / `--config`）；
  ② `export PYTHONPATH="$OPENSTL_DIR:$PYTHONPATH"`；
  ③ 启动前依赖预检（`torch/numpy/zarr/lightning/timm/fvcore`）与路径可读性检查，快速失败；
  ④ `RESUME=auto` 时自动把上一次的 `checkpoints/last.ckpt` 作为 `--ckpt_path` 续跑；
  ⑤ 每次运行落到 `work_dirs/<EX_NAME>/`，并发作业可加 `$SLURM_JOB_ID` 后缀避免覆盖。
- 注意：`fvcore` 是 `openstl/api/exp.py` 的**顶层 import**，即使 `--display-method-info`
  未开启也必须先装上。
- 状态：已修复（2026-09-30）。

## DBG-005 `import openstl` 的依赖链缺失与 timm 版本冲突
- 现象：按报错顺序依次出现 5 个失败，藏得很深（每补一个才暴露下一个）：

  | 报错 | 需补 | 引入位置 |
  | --- | --- | --- |
  | `No module named 'cv2'` | `opencv-python-headless` | `openstl/utils/main_utils.py` |
  | `No module named 'imageio'` | `imageio` | `openstl/utils/visualization.py` |
  | `cannot import name 'Nadam' from 'timm.optim.nadam'` | `timm==0.9.12`（1.0.30 已移除 Nadam） | `openstl/core/optim_scheduler.py` |
  | `No module named 'pywt'` | `PyWavelets` | `openstl/modules/wast_modules.py` |
  | `No module named 'skimage'` | `scikit-image` | `openstl/datasets/dataloader_kitticaltech.py` |

- 根因：`openstl/__init__.py` → `methods/__init__.py`（全量 method）→
  `datasets/__init__.py`（全量 loader）是**无条件全量导入**，因此即使只用 SimVP，
  也必须满足 WaST / KittiCaltech 等无关部件的依赖。
- 修正认知：此前「环境未就绪」的判断来自用 **base python** 探测；
  实际上依赖在 conda env `test` 里基本齐全，真正的缺口是 cv2/imageio/pywt/skimage
  以及 timm 版本。已按 `docs/dev/environment.md` 新建 `openstl` 环境解决。
- 状态：已修复（2026-09-30）；slurm 的依赖预检已同步扩充到这些包并加了 timm<1.0 断言。

## DBG-006 参考脚本 `test_dl/test_single_sample_Chase2025.py` 已被删除
- 现象：该脚本已从工作区删除（2026-09-30 由用户手动删除），但仍在 4 处被引用：
  `tools/train_simvp_goes13_single_sample.py`（docstring 与 `colorize` 注释）、
  `openstl/datasets/dataloader_goes13.py`（数据契约出处）、
  `docs/experiments/EXP-003.../README.md`、`docs/project/knowledge.md`。
- 影响：**仅文档/注释引用失效，不影响运行** —— 没有任何代码 import 它，
  其 `colorize()` 已就地复刻到 `tools/train_simvp_goes13_single_sample.py`。
- 遗留约定（已从该脚本固化到新脚本，不会丢失）：配色 `Spectral_r`、`vmin=-4 / vmax=2`、
  2 帧输入 → 1 帧输出、float16 → float32。
- 建议：如需再次对照，可从 git 历史或 `test_dl/test_dl.py` 方向找回；若确定不再复核，
  可把上述 4 处引用改为指向 `EXP-001-cira-diff-data-audit.md`。

## DBG-007 `preload=True` 全量训练内存爆炸
- 现象（预测风险，非已发生）：adapter 默认 `preload=True`，会把选中的子集**整个**读进内存。
  每样本 = 3 帧 × 256×256 × 4 字节 ≈ 0.79 MB；全量 35,595 样本 ≈ **28 GB**，必然 OOM。
- 解法：`tools/train_simvp_goes13_smoke.py` 在 `load_data` 前加了护栏，当
  `preload=True` 且总样本 > 4000 时打印 `[DBG-007][WARN] ...` 提醒加 `--no-preload`。
  全量训练请显式 `--no-preload`（改为逐条读 zarr，`create_loader` 的 `drop_last` 仍安全）。
- 关联：mean/std 统计（`_compute_mean_std`）是逐样本 Python 循环，全量 split 会很慢；
  规模化时应改成分块统计（已记入待办，暂未实现）。
- 状态：护栏已加（2026-09-30）。
