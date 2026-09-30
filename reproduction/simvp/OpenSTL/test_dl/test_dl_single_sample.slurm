#!/bin/bash
#SBATCH --job-name=simvp_goes13_single_sample
#SBATCH --output=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/single_sample_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/single_sample_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# SimVP 单样本过拟合验证（EXP-003）的提交脚本。
# 目的：在 ONE 样本上让 SimVP overfit 到收敛，验证模型本身接线正确
#       （通道对齐 / 时间轴 / skip 连接 / loss 算在正确张量上）。
#       聚合指标发现不了的接线错误，会在这里的并列图里立刻暴露。
#
# 关键约束见 docs/dev/debugging.md 的 DBG-003 / DBG-004 / DBG-005：
#   * 所有路径必须是绝对路径（slurm 的工作目录取决于提交位置）
#   * 必须把 OpenSTL 仓库根放进 PYTHONPATH，否则 import openstl 失败
#   * 必须用 openstl 环境（test 环境 timm>=1.0 会在 import 阶段崩，见 DBG-005）
#
# 与 test_dl.slurm（EXP-002 小样本流水线）的区别：
#   * 本脚本跑 tools/train_simvp_goes13_single_sample.py，没有 batch-size / n-train /
#     n-val / n-test / resume 这些概念，只有 --index / --ref-samples / --epochs。
#   * 单样本 preload=True 只占 1 条样本的内存，不会 OOM（无需 DBG-007 护栏）。

set -euo pipefail

# ----------------------------- 可按需修改 -----------------------------
CONDA_ENV=openstl
CONDA_BASE=/home/group1/miniconda3

OPENSTL_DIR=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL
LOGDIR="$OPENSTL_DIR/test_dl/logs"
ZARR=/home/group1/26fall_aiclass/yr/data1/data1/satcast/edm_GOES_ch13_train_dataset.zarr
CONFIG="$OPENSTL_DIR/configs/goes13/simvp/SimVP_gSTA.py"

# 产物根目录（绝对路径）。EXP-003 的 png/npy/json 写到这里，checkpoints 写到
# "$OUT_ROOT/$EX_NAME/"。
OUT_ROOT="$OPENSTL_DIR/work_dirs"
EX_NAME=${EX_NAME:-simvp_goes13_single_sample}
OUTPUT_DIR="$OUT_ROOT/single_sample_goes13_output"

# 以下变量都可在提交时用环境变量覆盖，例如：
#   INDEX=5 EPOCHS=50 sbatch test_dl/test_dl_single_sample.slurm
INDEX=${INDEX:-0}
EPOCHS=${EPOCHS:-100}
REF_SAMPLES=${REF_SAMPLES:-32}        # 归一化统计量的参考池大小（取自身会退化，见 EXP-003 D4）
# ---------------------------------------------------------------------

mkdir -p "$LOGDIR" "$OUTPUT_DIR"
cd "$OPENSTL_DIR"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[INFO] job id       : ${SLURM_JOB_ID:-N/A}"
echo "[INFO] node         : $(hostname)"
echo "[INFO] python       : $(which python)  ($(python -V 2>&1))"
echo "[INFO] overfit idx  : $INDEX   epochs=$EPOCHS   ref_samples=$REF_SAMPLES"
echo "[INFO] artifact dir : $OUTPUT_DIR"
nvidia-smi -L || echo "[WARN] nvidia-smi 不可用"

export PYTHONPATH="$OPENSTL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"
mkdir -p "$MPLCONFIGDIR"

# ------------------------- 依赖预检（快速失败） -------------------------
python -u - <<'PY'
import importlib, sys

CHECKED = [
    ("torch", "torch"), ("numpy", "numpy"), ("zarr", "zarr"),
    ("lightning", "lightning"), ("timm", "timm"), ("fvcore", "fvcore"),
    ("cv2", "opencv-python-headless"), ("imageio", "imageio"),
    ("pywt", "PyWavelets"), ("skimage", "scikit-image"),
]
missing = []
for module, package in CHECKED:
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append("  - {}: {}: {}".format(module, type(exc).__name__, exc))
if missing:
    sys.stderr.write(
        "[FATAL] 缺少依赖:\n" + "\n".join(missing)
        + "\n请执行: pip install " + " ".join(p[1] for p in CHECKED) + "\n")
    sys.exit(2)

import timm
if int(timm.__version__.split(".")[0]) >= 1:
    sys.stderr.write(
        "[FATAL] timm={} 不兼容（>= 1.0 移除了 timm.optim.nadam.Nadam）。\n"
        "请执行: pip install timm==0.9.12\n".format(timm.__version__))
    sys.exit(2)

import torch
print("[OK] deps ok | torch={} | timm={} | cuda_available={} | device_count={}".format(
    torch.__version__, timm.__version__, torch.cuda.is_available(), torch.cuda.device_count()))
PY

python -u -c "import openstl, os; print('[OK] openstl from', os.path.dirname(openstl.__file__))"

[ -d "$ZARR" ] || { echo "[FATAL] zarr 不存在: $ZARR"; exit 3; }
[ -f "$CONFIG" ] || { echo "[FATAL] config 不存在: $CONFIG"; exit 3; }
[ -w "$OUTPUT_DIR" ] || { echo "[FATAL] 产物目录不可写: $OUTPUT_DIR"; exit 3; }

python -u "$OPENSTL_DIR/tools/train_simvp_goes13_single_sample.py" \
    --zarr "$ZARR" \
    --config "$CONFIG" \
    --index "$INDEX" \
    --ref-samples "$REF_SAMPLES" \
    --epochs "$EPOCHS" \
    --model-type gSTA \
    --output-dir "$OUTPUT_DIR" \
    --res_dir "$OUT_ROOT" \
    --ex_name "$EX_NAME"

echo "[DONE] artifacts: $OUTPUT_DIR"
echo "[HINT] 判读：loss 应下降若干数量级，predicted 面板应贴近 truth（error 图很小）。"
echo "       若 loss 平坦或 prediction 无结构，说明模型/接线有问题，而非数据难。"
