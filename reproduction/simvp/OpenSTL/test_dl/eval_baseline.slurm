#!/bin/bash
#SBATCH --job-name=simvp_baseline_eval
#SBATCH --output=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/eval_baseline_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/eval_baseline_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# SimVP 的 baseline 对齐评估（统一协议 v1，见 docs/experiments/EXP-004）。
# 与 test_dl.slurm（EXP-002 训练）同构：绝对路径 + 独占一张卡 + openstl 环境 + PYTHONPATH。
#
# 环境变量均可覆盖，例如：
#   ROLLOUT=18 sbatch test_dl/eval_baseline.slurm            # 3 小时 rollout（对齐 ly）
#   CKPT=.../best.ckpt NORM_MEAN=0 NORM_STD=1 sbatch ...     # 按 baseline 协议新训的模型
#
# 重要：评估的是哪个 checkpoint 就用它**训练时的归一化常数**：
#   * EXP-002 的 smoke ckpt（子集自算）  -> NORM_MEAN=0.093006 NORM_STD=0.865920
#   * 按 baseline 协议新训的（吃 zarr 原值）-> NORM_MEAN=0 NORM_STD=1

set -euo pipefail

# ----------------------------- 可按需修改 -----------------------------
CONDA_ENV=openstl
CONDA_BASE=/home/group1/miniconda3

OPENSTL_DIR=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL
LOGDIR="$OPENSTL_DIR/test_dl/logs"
DATA_ROOT=/home/group1/26fall_aiclass/yr/data1/data1/satcast
ZARR=edm_GOES_ch13_test_dataset.zarr

CKPT=${CKPT:-$OPENSTL_DIR/work_dirs/simvp_goes13_smoke/checkpoints/best.ckpt}
CONFIG=${CONFIG:-$OPENSTL_DIR/configs/goes13/simvp/SimVP_gSTA.py}
OUT_DIR=${OUT_DIR:-$OPENSTL_DIR/work_dirs/baseline_eval}

# 1 = 仅单步(t+10min)；18 = 对齐 ly 的 3 小时 rollout
ROLLOUT=${ROLLOUT:-1}
BATCH_SIZE=${BATCH_SIZE:-16}

# 老 smoke ckpt 用子集自算的归一化常数；新训模型改成 0 / 1
NORM_MEAN=${NORM_MEAN:-0.093006}
NORM_STD=${NORM_STD:-0.865920}
# ---------------------------------------------------------------------

mkdir -p "$LOGDIR" "$OUT_DIR"
cd "$OPENSTL_DIR"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[INFO] job id    : ${SLURM_JOB_ID:-N/A}"
echo "[INFO] node      : $(hostname)"
echo "[INFO] python    : $(which python)  ($(python -V 2>&1))"
echo "[INFO] ckpt      : $CKPT"
echo "[INFO] rollout   : $ROLLOUT 步 (=$((ROLLOUT * 10)) 分钟)"
echo "[INFO] norm      : mean=$NORM_MEAN std=$NORM_STD"
echo "[INFO] out dir   : $OUT_DIR"
nvidia-smi -L || echo "[WARN] nvidia-smi 不可用"

export PYTHONPATH="$OPENSTL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"
mkdir -p "$MPLCONFIGDIR"

[ -f "$CKPT" ] || { echo "[FATAL] checkpoint 不存在: $CKPT"; exit 3; }
[ -f "$CONFIG" ] || { echo "[FATAL] config 不存在: $CONFIG"; exit 3; }
[ -d "$DATA_ROOT/$ZARR" ] || { echo "[FATAL] zarr 不存在: $DATA_ROOT/$ZARR"; exit 3; }

python -u "$OPENSTL_DIR/tools/eval_simvp_baseline.py" \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT" \
    --zarr "$ZARR" \
    --out_dir "$OUT_DIR" \
    --rollout-steps "$ROLLOUT" \
    --batch_size "$BATCH_SIZE" \
    --norm-mean "$NORM_MEAN" \
    --norm-std "$NORM_STD"

echo "[DONE] metrics: $OUT_DIR/baseline_metrics.json"
echo "[HINT] 与 baseline 对照：UNet 单步 MSE≈0.00809 (RMSE≈1.74 K)；"
echo "       SimVP(cb 全量) RMSE≈1.57 K；persistence≈4.55 K。"
