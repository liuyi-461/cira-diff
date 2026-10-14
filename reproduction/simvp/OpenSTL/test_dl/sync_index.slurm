#!/bin/bash
#SBATCH --job-name=sync_index
#SBATCH --output=test_dl/logs/sync_index_%j.out
#SBATCH --error=test_dl/logs/sync_index_%j.err
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
# 可移植性：仓库根按优先级取 OPENSTL_DIR > SLURM_SUBMIT_DIR（校验含 tools/ 与 openstl/）> 内置默认值。
# ⚠️ 不能用 ${BASH_SOURCE[0]} 推导：SLURM 会把脚本复制到 /var/spool/slurmd/jobNNNN/slurm_script
#    再执行，BASH_SOURCE 指向 spool 而非原路径（实测会推导出 /var/spool/slurmd，已踩坑）。
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

# 护栏：解析出的仓库根必须含 tools/ 与配置文件，否则立即失败。
# 目的：把"路径解析错却静默跑歪"变成启动即报错（曾在 SLURM spool 路径上踩坑）。
if [ ! -d "$OPENSTL_DIR/tools" ] || [ ! -f "$OPENSTL_DIR/configs/satcast/SimVP.py" ]; then
    echo "[FATAL] 仓库根解析失败：$OPENSTL_DIR" >&2
    echo "[FATAL] 缺 tools/ 或 configs/satcast/SimVP.py。" >&2
    echo "[FATAL] 请从 OpenSTL 根目录执行 sbatch test_dl/xxx.slurm，或显式 export OPENSTL_DIR=<仓库根>" >&2
    exit 3
fi
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
