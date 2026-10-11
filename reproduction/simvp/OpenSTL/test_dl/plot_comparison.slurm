#!/bin/bash
#SBATCH --job-name=plot_comparison
#SBATCH --output=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/plot_comparison_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/plot_comparison_%j.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#
# 生成 SimVP(多 seed) vs UNet(ly, 3 seed) 的逐时效对比图（纯绘图，无需 GPU）。
# 等三 seed 训练汇总(238)完成后自动执行。
#
# 提交方式（作业号会变，提交时用 --dependency 覆盖）：
#   sbatch --dependency=afterany:238 test_dl/plot_comparison.slurm

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
python -u "$OPENSTL_DIR/tools/plot_unet_comparison.py"
echo "[DONE] 对比图已生成：docs/experiments/EXP-005-simvp-multiseed/unet_vs_simvp_rollout.png"
