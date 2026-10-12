#!/bin/bash
#SBATCH --job-name=eval_arms
#SBATCH --output=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/eval_arms_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/eval_arms_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# 两个对照臂训练完成后自动：
#   1) 用统一协议评估（test 全量 1024 条 × 18 步）
#   2) 导出为 pred_official.npy 同格式 (1024,18,256,256)
#   3) 四臂统一对比（baseline / GAN / 多步 p=0 / 多步+SS p=0.5）并回填 EXP-006
#
# 提交：
#   sbatch --dependency=afterany:280:281 test_dl/eval_rollout_arms.slurm

set -euo pipefail

CONDA_ENV=openstl
CONDA_BASE=/home/group1/miniconda3
OPENSTL_DIR=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL
LOGDIR="$OPENSTL_DIR/test_dl/logs"
DATA_ROOT=/home/group1/26fall_aiclass/yr/data1/data1/satcast
CONFIG="$OPENSTL_DIR/configs/satcast/SimVP.py"

mkdir -p "$LOGDIR"
cd "$OPENSTL_DIR"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$OPENSTL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"
mkdir -p "$MPLCONFIGDIR"

echo "[INFO] job id : ${SLURM_JOB_ID:-N/A}"

# ---------- 1) 评估两个对照臂 ----------
for D in satcast_simvp_rollout_ss0.0 satcast_simvp_rollout_ss0.5 satcast_simvp_rollout_ss05_full; do
    CKPT="$OPENSTL_DIR/work_dirs/$D/checkpoints/best.ckpt"
    if [ ! -f "$CKPT" ]; then
        echo "[WARN] 跳过 $D（无 best.ckpt，训练可能未完成）"
        continue
    fi
    echo "================= 评估 $D（1024 条 × 18 步） ================="
    python -u "$OPENSTL_DIR/tools/eval_simvp_baseline.py" \
        --ckpt "$CKPT" --config "$CONFIG" --data_root "$DATA_ROOT" \
        --out_dir "$OPENSTL_DIR/work_dirs/$D/baseline_eval_rollout18_full" \
        --rollout-steps 18 --max-samples 1024 --batch_size 16 \
        --norm-mean 0 --norm-std 1
done

# ---------- 2) 导出为官方格式 ----------
echo "================= 导出 npy（1024,18,256,256） ================="
python -u "$OPENSTL_DIR/tools/export_preds_official_format.py" \
    --extra rollout_ss0.0=satcast_simvp_rollout_ss0.0 \
            rollout_ss0.5=satcast_simvp_rollout_ss0.5 \
            rollout_ss05_full=satcast_simvp_rollout_ss05_full \
            gan=satcast_simvp_gan

# ---------- 3) 四臂统一对比并回填 EXP-006 ----------
echo "================= 四臂统一对比 ================="
python -u "$OPENSTL_DIR/tools/compare_arms.py"

echo "[DONE] 对照实验完成，见 docs/experiments/EXP-006-anti-blur/"
