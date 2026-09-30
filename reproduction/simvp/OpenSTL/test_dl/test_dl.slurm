#!/bin/bash
#SBATCH --job-name=simvp_goes13_smoke
#SBATCH --output=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/test_dl_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/test_dl_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# SimVP 小样本流水线验证（EXP-002）的提交脚本。
# 关键约束见 docs/dev/debugging.md 的 DBG-003 / DBG-004：
#   * 所有路径必须是绝对路径（slurm 的工作目录取决于提交位置）
#   * 必须把 OpenSTL 仓库根放进 PYTHONPATH，否则 import openstl 失败

set -euo pipefail

# ----------------------------- 可按需修改 -----------------------------
# 必须用 openstl 环境：它是在 test 环境基础上克隆后把 timm 钉到 0.9.12 得到的
# （timm >= 1.0 删除了 timm.optim.nadam.Nadam，openstl.core.optim_scheduler 会 ImportError）。
# 直接用 test 环境会在 import openstl 阶段就失败，见 docs/dev/debugging.md 的 DBG-005。
CONDA_ENV=openstl
CONDA_BASE=/home/group1/miniconda3

OPENSTL_DIR=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL
LOGDIR="$OPENSTL_DIR/test_dl/logs"
ZARR=/home/group1/26fall_aiclass/yr/data1/data1/satcast/edm_GOES_ch13_train_dataset.zarr
CONFIG="$OPENSTL_DIR/configs/goes13/simvp/SimVP_gSTA.py"

# 产物根目录（绝对路径）。程序会在 trainer.fit() 开始时自行创建
# "$OUT_ROOT/$EX_NAME/" 与 "…/checkpoints/"，无需手动 mkdir。
OUT_ROOT="$OPENSTL_DIR/work_dirs"
EX_NAME=${EX_NAME:-simvp_goes13_smoke}
# 若需并发提交多个作业，取消下一行注释让每次运行落到独立目录，避免互相覆盖。
# EX_NAME="${EX_NAME}_${SLURM_JOB_ID}"
RUN_DIR="$OUT_ROOT/$EX_NAME"

# 以下变量都可在提交时用环境变量覆盖，例如：
#   EPOCHS=1 N_TRAIN=32 N_VAL=8 N_TEST=8 sbatch test_dl/test_dl.slurm
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-8}
N_TRAIN=${N_TRAIN:-128}
N_VAL=${N_VAL:-32}
N_TEST=${N_TEST:-32}
NUM_WORKERS=${NUM_WORKERS:-2}      # 需 <= cpus-per-task - 1

# auto = 若上一次的 last.ckpt 存在则续跑（被超时杀掉时用）；off = 从头训练
RESUME=${RESUME:-auto}
# ---------------------------------------------------------------------

mkdir -p "$LOGDIR"
cd "$OPENSTL_DIR"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[INFO] job id       : ${SLURM_JOB_ID:-N/A}"
echo "[INFO] node         : $(hostname)"
echo "[INFO] python       : $(which python)  ($(python -V 2>&1))"
echo "[INFO] workdir      : $(pwd)"
echo "[INFO] artifact dir : $RUN_DIR"
nvidia-smi -L || echo "[WARN] nvidia-smi 不可用"

# openstl 未安装到 site-packages：Python 只把「脚本所在目录」(…/OpenSTL/tools)
# 放进 sys.path，所以必须显式把仓库根加入 PYTHONPATH，否则 import openstl 失败。
export PYTHONPATH="$OPENSTL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"
mkdir -p "$MPLCONFIGDIR"

# ------------------------- 依赖预检（快速失败） -------------------------
python -u - <<'PY'
import importlib, sys

# 这些都是 import openstl 这条链路上的**顶层**依赖，缺任何一个都会在 import 阶段就崩：
#   cv2      <- openstl/utils/main_utils.py
#   imageio  <- openstl/utils/visualization.py
#   fvcore   <- openstl/api/exp.py（即使不打印 FLOPs 也必须装）
#   pywt     <- openstl/modules/wast_modules.py
#   skimage  <- openstl/datasets/dataloader_kitticaltech.py
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

# timm >= 1.0 移除了 Nadam，而 openstl.core.optim_scheduler 需要它。
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
[ -w "$OUT_ROOT" ] || [ -w "$OPENSTL_DIR" ] || { echo "[FATAL] 产物目录不可写: $OUT_ROOT"; exit 3; }

# -------------------------------- 续跑 --------------------------------
CKPT_ARG=()
CKPT_FILE="$RUN_DIR/checkpoints/last.ckpt"
if [ "$RESUME" = "auto" ] && [ -f "$CKPT_FILE" ]; then
    echo "[INFO] resume from $CKPT_FILE"
    CKPT_ARG=(--ckpt_path "$CKPT_FILE")
fi

python -u "$OPENSTL_DIR/tools/train_simvp_goes13_smoke.py" \
    --zarr "$ZARR" \
    --config "$CONFIG" \
    --res_dir "$OUT_ROOT" \
    --ex_name "$EX_NAME" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --n-train "$N_TRAIN" \
    --n-val "$N_VAL" \
    --n-test "$N_TEST" \
    --num-workers "$NUM_WORKERS" \
    ${CKPT_ARG[@]+"${CKPT_ARG[@]}"}

echo "[DONE] artifacts: $RUN_DIR"
echo "[HINT] 详细超参与环境信息在: $RUN_DIR/model_param.json 和 $RUN_DIR/train_*.log"
