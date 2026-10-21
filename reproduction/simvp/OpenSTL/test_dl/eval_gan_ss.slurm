#!/bin/bash
#SBATCH --job-name=eval_gan_ss
#SBATCH --output=test_dl/logs/eval_gan_ss_%j.out
#SBATCH --error=test_dl/logs/eval_gan_ss_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# GAN+SS 臂的统一协议评估（test 1024 条 × 18 步）+ 导出官方格式 npy + 更新多臂对比。
#
# 提交：sbatch --dependency=afterany:<训练jobid> test_dl/eval_gan_ss.slurm

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-openstl}
CONDA_BASE=${CONDA_BASE:-/home/group1/miniconda3}
# 可移植性：仓库根由 SLURM_SUBMIT_DIR 推导（BASH_SOURCE 在 SLURM 中指向 spool，不可用）
OPENSTL_DIR="${OPENSTL_DIR:-}"
if [ -z "$OPENSTL_DIR" ]; then
    for _cand in "${SLURM_SUBMIT_DIR:-}" "$(pwd)"; do
        if [ -n "$_cand" ] && [ -d "$_cand/tools" ] && [ -d "$_cand/openstl" ]; then
            OPENSTL_DIR="$_cand"
            break
        fi
    done
fi
OPENSTL_DIR="${OPENSTL_DIR:-/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL}"
if [ ! -d "$OPENSTL_DIR/tools" ] || [ ! -f "$OPENSTL_DIR/configs/satcast/SimVP.py" ]; then
    echo "[FATAL] 仓库根解析失败：$OPENSTL_DIR" >&2
    exit 3
fi
LOGDIR="$OPENSTL_DIR/test_dl/logs"
DATA_ROOT="${SATCAST_ROOT:-/home/group1/26fall_aiclass/yr/data1/data1/satcast}"
CONFIG="$OPENSTL_DIR/configs/satcast/SimVP.py"

# ----------------------------- 可调参数 -----------------------------
EX_NAME=${EX_NAME:-satcast_simvp_gan_ss}
MAX_SAMPLES=${MAX_SAMPLES:-1024}   # 1024 = 全量（配对比检验与逐样本图用）
ROLLOUT=${ROLLOUT:-18}
# -------------------------------------------------------------------

mkdir -p "$LOGDIR"; cd "$OPENSTL_DIR"
source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
export PYTHONPATH="$OPENSTL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"; mkdir -p "$MPLCONFIGDIR"

echo "[INFO] job id : ${SLURM_JOB_ID:-N/A}"
OUT_DIR="$OPENSTL_DIR/work_dirs/$EX_NAME/baseline_eval_rollout18_full"
CKPT="$OPENSTL_DIR/work_dirs/$EX_NAME/checkpoints/best.ckpt"

if [ ! -f "$CKPT" ]; then
    echo "[FATAL] 未找到 checkpoint：$CKPT（训练可能未完成或失败）"
    exit 3
fi

echo "================= 1) 统一协议评估（test $MAX_SAMPLES 条 × $ROLLOUT 步） ================="
python -u "$OPENSTL_DIR/tools/eval_simvp_baseline.py" \
    --ckpt "$CKPT" --config "$CONFIG" --data_root "$DATA_ROOT" \
    --out_dir "$OUT_DIR" \
    --rollout-steps "$ROLLOUT" --max-samples "$MAX_SAMPLES" --batch_size 16 \
    --norm-mean 0 --norm-std 1

echo "================= 2) 导出官方格式 npy（1024,18,256,256） ================="
python -u "$OPENSTL_DIR/tools/export_preds_official_format.py" \
    --extra gan_ss="$EX_NAME"

echo "================= 3) 更新多臂对比（含新增 GAN+SS 臂） ================="
python -u "$OPENSTL_DIR/tools/compare_arms.py" || \
    echo "[WARN] compare_arms.py 失败（可能尚未把新臂加入 ARMS 列表），不影响评估与导出"

echo "[DONE] GAN+SS 评估完成：$OUT_DIR"
