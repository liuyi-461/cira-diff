#!/bin/bash
#SBATCH --job-name=compare_1024
#SBATCH --output=test_dl/logs/compare_1024_%j.out
#SBATCH --error=test_dl/logs/compare_1024_%j.err
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
