#!/bin/bash
#SBATCH --job-name=compare_1024
#SBATCH --output=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/compare_1024_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/compare_1024_%j.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# 1024 条 rollout 评估(264)完成后：
#   1) 重跑 SimVP vs UNet 的并排对比 + 配对检验（N=1024）
#   2) 自动把结果回填到 EXP-005 README 的 PAIRED 标记区
#
# 提交：sbatch --dependency=afterany:264 test_dl/compare_1024.slurm

set -euo pipefail

CONDA_ENV=openstl
CONDA_BASE=/home/group1/miniconda3
OPENSTL_DIR=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL
LOGDIR="$OPENSTL_DIR/test_dl/logs"

mkdir -p "$LOGDIR"
cd "$OPENSTL_DIR"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$OPENSTL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"
mkdir -p "$MPLCONFIGDIR"

echo "[INFO] job id : ${SLURM_JOB_ID:-N/A}"

# 校验三个 seed 的 1024 条 rollout 结果是否齐全
MISS=0
for D in satcast_simvp_full satcast_simvp_full_seed0 satcast_simvp_full_seed123; do
    F="$OPENSTL_DIR/work_dirs/$D/baseline_eval_rollout18_full/preds.npy"
    if [ ! -f "$F" ]; then
        echo "[WARN] 缺少 $F"
        MISS=1
    fi
done
if [ "$MISS" = "1" ]; then
    echo "[FATAL] 1024 条 rollout 结果不完整，终止（不覆盖已有文档）"
    exit 3
fi

python -u "$OPENSTL_DIR/tools/compare_with_unet_raw.py" \
    --n-samples 1024 \
    --pred-subdir baseline_eval_rollout18_full

echo "[DONE] 1024 条配对检验完成，EXP-005 README 已回填"
