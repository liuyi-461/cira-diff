#!/bin/bash
#SBATCH --job-name=simvp_multiseed
#SBATCH --output=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/aggregate_multiseed_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/aggregate_multiseed_%j.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#
# 多 seed 汇总（纯读 JSON + 写文档，**不需要 GPU**）。
# 等三个 seed 的训练都结束后自动跑，生成 EXP-005 实验卡。
#
# 提交方式（依赖三个训练作业；作业号会变，提交时用 --dependency 覆盖）：
#   sbatch --dependency=afterany:214:236:237 test_dl/aggregate_multiseed.slurm

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

echo "[INFO] job id : ${SLURM_JOB_ID:-N/A}"
echo "[INFO] 汇总三个 seed：42 / 0 / 123"

python -u "$OPENSTL_DIR/tools/aggregate_multiseed.py"

echo "[DONE] EXP-005 已生成：docs/experiments/EXP-005-simvp-multiseed/README.md"
