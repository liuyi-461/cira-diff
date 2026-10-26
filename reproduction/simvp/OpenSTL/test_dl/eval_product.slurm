#!/bin/bash
#SBATCH --job-name=evalprod
#SBATCH --output=test_dl/logs/eval_product_%j.out
#SBATCH --error=test_dl/logs/eval_product_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#
# 产品评估：生成 18 步 test 预测 + 跑 8 指标（BT PDF / 径向 PSD / SSIM / CSI / FSS /
# CRPS / 覆盖率 / 云团移动 / 冷却率 / 面积增长 / LapVar）。
#
# 环境变量：
#   A_CKPT   产品 A 权重（必填）
#   B_CKPT   产品 B 权重；传 none 则只评估 A 单独（作对照）
#   EX_NAME  产品 B 的 ex-name（B_CKPT=none 时仅用于命名，可随便给）
#   TAG      评估输出目录名（docs/experiments/EXP-008-metrics-baseline/<TAG>/）
#
# 例：
#   A_CKPT=.../productA_gan_k6/checkpoints/best.ckpt B_CKPT=none \
#       EX_NAME=Aonly TAG=A_gan sbatch test_dl/eval_product.slurm
#   A_CKPT=.../productA_gan_k6/checkpoints/best.ckpt \
#       B_CKPT=.../productB_gan_k6/checkpoints/best.pt \
#       EX_NAME=productB_gan_k6 TAG=AB_gan sbatch test_dl/eval_product.slurm

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-openstl}
CONDA_BASE=${CONDA_BASE:-/home/group1/miniconda3}

OPENSTL_DIR="${OPENSTL_DIR:-}"
if [ -z "$OPENSTL_DIR" ]; then
    for _cand in "${SLURM_SUBMIT_DIR:-}" "$(pwd)"; do
        if [ -n "$_cand" ] && [ -d "$_cand/tools" ] && [ -d "$_cand/openstl" ]; then
            OPENSTL_DIR="$_cand"; break
        fi
    done
fi
OPENSTL_DIR="${OPENSTL_DIR:-/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL}"
if [ ! -d "$OPENSTL_DIR/tools" ]; then
    echo "[FATAL] 仓库根解析失败：$OPENSTL_DIR" >&2; exit 3
fi

A_CKPT="${A_CKPT:?必须给 A_CKPT}"
B_CKPT="${B_CKPT:-none}"
EX_NAME="${EX_NAME:-Aonly}"
TAG="${TAG:-${EX_NAME}}"
LEADS="${LEADS:-1,2,3,6,9,12,15,18}"

LOGDIR="$OPENSTL_DIR/test_dl/logs"
mkdir -p "$LOGDIR"
cd "$OPENSTL_DIR"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$OPENSTL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8

OUTDIR="$OPENSTL_DIR/docs/experiments/EXP-008-metrics-baseline/$TAG"
# ⚠️ 预文件路径必须含 TAG 且唯一：曾用 $EX_NAME 作路径，导致两路评估
#    （EX_NAME 都取 Aonly）写到同一文件，互相覆盖 + 读写竞态，
#    最终两路指标完全相同且 RMSE 异常（9.999）。按 TAG 隔离后可并行。
PREDDIR="$OPENSTL_DIR/work_dirs/eval_preds/$TAG"
PREDS="$PREDDIR/preds.npy"

echo "[INFO] job=${SLURM_JOB_ID:-N/A} node=$(hostname)"
echo "[INFO] A_CKPT = $A_CKPT"
echo "[INFO] B_CKPT = $B_CKPT   ($([ "$B_CKPT" = "none" ] && echo '只评估 A 单独' || echo '最终产品 A+B'))"
echo "[INFO] TAG=$TAG  LEADS=$LEADS"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | head -1 || true
[ -f "$A_CKPT" ] || { echo "[FATAL] 缺 A 权重：$A_CKPT"; exit 3; }
if [ "$B_CKPT" != "none" ]; then
  [ -f "$B_CKPT" ] || { echo "[FATAL] 缺 B 权重：$B_CKPT"; exit 3; }
fi

mkdir -p "$PREDDIR" "$OUTDIR"

echo "=== 步骤 1/2：生成 18 步 test 预测 ==="
python -u "$OPENSTL_DIR/tools/make_final_preds.py" \
    --config "$OPENSTL_DIR/configs/satcast/SimVP.py" \
    --ex-name "$EX_NAME" \
    --a-ckpt "$A_CKPT" \
    --b-ckpt "$B_CKPT" \
    --steps 18 --batch 16 \
    ${B_LEADS:+--b-leads "$B_LEADS"} \
    ${B_DECAY:+--b-decay} \
    --out "$PREDS"

echo "=== 步骤 2/2：8 指标评估 ==="
python -u "$OPENSTL_DIR/tools/eval_metrics.py" \
    --preds "$PREDS" \
    --out-dir "$OUTDIR" \
    --leads "$LEADS" \
    --max-samples 1024 --batch 16

echo "[DONE] 预测 $PREDS"
echo "[DONE] 指标 $OUTDIR"
