#!/bin/bash
#SBATCH --job-name=simvp_satcast_full
#SBATCH --output=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/train_baseline_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/train_baseline_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# SimVP 全量训练（baseline 对齐版）+ 训练后自动按统一协议评估。
#
# 对齐要点（详见 docs/experiments/EXP-004-simvp-vs-baseline/README.md）：
#   * train / validation / test 各读一份 zarr，validation/test 是真正的 held-out
#   * 不做子集 z-score：zarr 已是归一化空间，模型直接吃原值 -> 评估用 --norm-mean 0 --norm-std 1
#   * 超参全部来自 configs/satcast/SimVP.py（唯一来源），本脚本不重复定义
#
# 环境变量覆盖示例：
#   SMOKE=1 sbatch test_dl/train_baseline.slurm     # 冒烟：少量样本 2 epoch，产物独立目录
#   RESUME=0 sbatch test_dl/train_baseline.slurm    # 改了超参后强制从头训
#   EPOCHS=5 sbatch test_dl/train_baseline.slurm    # 只训 5 epoch（试跑）
#   ROLLOUT=0 sbatch test_dl/train_baseline.slurm   # 训练后只跑单步评估，不做 18 步 rollout

set -euo pipefail

# ----------------------------- 可按需修改 -----------------------------
CONDA_ENV=openstl
CONDA_BASE=/home/group1/miniconda3

OPENSTL_DIR=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL
LOGDIR="$OPENSTL_DIR/test_dl/logs"
DATA_ROOT=/home/group1/26fall_aiclass/yr/data1/data1/satcast
CONFIG="$OPENSTL_DIR/configs/satcast/SimVP.py"

EX_NAME=${EX_NAME:-satcast_simvp_full}
RES_DIR="${RES_DIR:-$OPENSTL_DIR/work_dirs}"
RUN_DIR="$RES_DIR/$EX_NAME"

# auto = 若存在 checkpoints/last.ckpt 就续跑（被 48h 时限杀掉时直接重提即可）；off = 从头训
RESUME=${RESUME:-auto}
# 训练后评估：1=单步；18=再跑一次 18 步 rollout（3 小时，对齐 ly 的 UNet）
ROLLOUT=${ROLLOUT:-18}
# 冒烟：1=只取少量样本 2 epoch，用于提交长任务前确认通路
SMOKE=${SMOKE:-0}
# 覆盖 epoch（不设则用 config 的 100）
EPOCHS=${EPOCHS:-}
# ---------------------------------------------------------------------

mkdir -p "$LOGDIR"
cd "$OPENSTL_DIR"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[INFO] job id    : ${SLURM_JOB_ID:-N/A}"
echo "[INFO] node      : $(hostname)"
echo "[INFO] python    : $(which python)  ($(python -V 2>&1))"
echo "[INFO] ex_name   : $EX_NAME"
echo "[INFO] run dir   : $RUN_DIR"
echo "[INFO] resume    : $RESUME   smoke=$SMOKE   epochs=${EPOCHS:-<config=100>}"
nvidia-smi -L || echo "[WARN] nvidia-smi 不可用"

export PYTHONPATH="$OPENSTL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"
mkdir -p "$MPLCONFIGDIR"

# ---------------------------- 前置校验（快速失败） ----------------------------
[ -f "$CONFIG" ] || { echo "[FATAL] config 不存在: $CONFIG"; exit 3; }
[ -d "$DATA_ROOT/edm_GOES_ch13_train_dataset.zarr" ] || { echo "[FATAL] train zarr 不存在: $DATA_ROOT"; exit 3; }
[ -d "$DATA_ROOT/edm_GOES_ch13_validation_dataset.zarr" ] || { echo "[FATAL] val zarr 不存在: $DATA_ROOT"; exit 3; }
[ -d "$DATA_ROOT/edm_GOES_ch13_test_dataset.zarr" ] || { echo "[FATAL] test zarr 不存在: $DATA_ROOT"; exit 3; }
[ -w "$RES_DIR" ] || [ -w "$OPENSTL_DIR" ] || { echo "[FATAL] 产物目录不可写: $RES_DIR"; exit 3; }

python -u -c "import torch, lightning, timm, fvcore, zarr; print('[OK] deps | torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# ------------------------------- 训练参数 -------------------------------
TRAIN_ARGS=()
if [ "$SMOKE" = "1" ]; then
    EX_NAME="${EX_NAME}_smoke"
    RUN_DIR="$RES_DIR/$EX_NAME"
    TRAIN_ARGS+=(--epochs 2 --limit-train 16 --limit-val 8 --limit-test 8)
    echo "[INFO] *** 冒烟模式：少量样本 2 epoch，产物写到 $RUN_DIR ***"
elif [ -n "$EPOCHS" ]; then
    TRAIN_ARGS+=(--epochs "$EPOCHS")
fi

CKPT_ARGS=()
LAST_CKPT="$RUN_DIR/checkpoints/last.ckpt"
if [ "$RESUME" = "auto" ] && [ -f "$LAST_CKPT" ]; then
    echo "[INFO] 发现 $LAST_CKPT，从该 checkpoint 续跑"
    CKPT_ARGS+=(--ckpt_path "$LAST_CKPT")
fi

# --------------------------------- 训练 ---------------------------------
python -u "$OPENSTL_DIR/tools/train_simvp_satcast.py" \
    --config "$CONFIG" \
    --res_dir "$RES_DIR" \
    --ex_name "$EX_NAME" \
    "${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}" \
    "${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}"

echo "[DONE] training artifacts: $RUN_DIR"

# ------------------------- 训练后按统一协议评估 -------------------------
# 权威指标来自这里（best.ckpt），不要用训练里的 exp.test()。
# 本脚本训出的模型直接吃 zarr 原值，故用 --norm-mean 0 --norm-std 1。
BEST_CKPT="$RUN_DIR/checkpoints/best.ckpt"
if [ ! -f "$BEST_CKPT" ]; then
    echo "[WARNING] 未找到 $BEST_CKPT，跳过评估"
    exit 0
fi

echo "[INFO] 单步评估（test 集 1024 条，对齐 baseline）"
python -u "$OPENSTL_DIR/tools/eval_simvp_baseline.py" \
    --ckpt "$BEST_CKPT" \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT" \
    --out_dir "$RUN_DIR/baseline_eval" \
    --rollout-steps 1 \
    --norm-mean 0 --norm-std 1 || echo "[WARNING] 单步评估失败（训练本身已成功）"

if [ "$ROLLOUT" = "18" ]; then
    echo "[INFO] 18 步 rollout 评估（3 小时，对齐 ly 的 UNet LT=1..18）"
    python -u "$OPENSTL_DIR/tools/eval_simvp_baseline.py" \
        --ckpt "$BEST_CKPT" \
        --config "$CONFIG" \
        --data_root "$DATA_ROOT" \
        --out_dir "$RUN_DIR/baseline_eval_rollout18" \
        --rollout-steps 18 \
        --norm-mean 0 --norm-std 1 || echo "[WARNING] rollout 评估失败（训练本身已成功）"
fi

echo "[DONE] metrics: $RUN_DIR/baseline_eval/baseline_metrics.json"
echo "[HINT] 对照：UNet 单步 MSE≈0.00809 (RMSE≈1.74 K)；SimVP(cb) RMSE≈1.57 K；persistence≈4.55 K"
