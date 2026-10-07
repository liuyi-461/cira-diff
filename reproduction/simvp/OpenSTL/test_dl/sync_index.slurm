#!/bin/bash
#SBATCH --job-name=sync_index
#SBATCH --output=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/sync_index_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/sync_index_%j.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#
# 把 EXP-004 / EXP-005 的结论同步到实验索引 docs/research/experiments.md。
# 纯读 JSON + 写文档，**不需要 GPU**。
#
# 提交方式（依赖多 seed 汇总作业；作业号会变，提交时用 --dependency 覆盖）：
#   sbatch --dependency=afterany:238 test_dl/sync_index.slurm

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
python -u "$OPENSTL_DIR/tools/sync_research_index.py"
echo "[DONE] 索引已同步：docs/research/experiments.md"
