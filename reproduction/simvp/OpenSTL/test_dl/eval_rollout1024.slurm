#!/bin/bash
#SBATCH --job-name=eval_rollout1024
#SBATCH --output=test_dl/logs/eval_rollout1024_%j.out
#SBATCH --error=test_dl/logs/eval_rollout1024_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# 把 rollout 评估从 128 条扩展到 **test 全量 1024 条**（三个 seed 各跑一次）。
# 目的：128 条时配对检验的标准误偏大；1024 条可显著提升检验功效。
# 产物写到各 seed 目录下的 baseline_eval_rollout18_full/，不覆盖原有的 128 条结果。

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
# 可移植性：数据根优先取环境变量 SATCAST_ROOT
DATA_ROOT="${SATCAST_ROOT:-/home/group1/26fall_aiclass/yr/data1/data1/satcast}"
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
df -h "$OPENSTL_DIR/work_dirs" | tail -1

for SEED_DIR in satcast_simvp_full satcast_simvp_full_seed0 satcast_simvp_full_seed123; do
    CKPT="$OPENSTL_DIR/work_dirs/$SEED_DIR/checkpoints/best.ckpt"
    OUT="$OPENSTL_DIR/work_dirs/$SEED_DIR/baseline_eval_rollout18_full"
    if [ ! -f "$CKPT" ]; then
        echo "[WARN] 跳过 $SEED_DIR（无 best.ckpt）"
        continue
    fi
    echo "================= $SEED_DIR（1024 条 × 18 步） ================="
    python -u "$OPENSTL_DIR/tools/eval_simvp_baseline.py" \
        --ckpt "$CKPT" \
        --config "$CONFIG" \
        --data_root "$DATA_ROOT" \
        --out_dir "$OUT" \
        --rollout-steps 18 \
        --max-samples 1024 \
        --batch_size 16 \
        --norm-mean 0 --norm-std 1
done

echo "[DONE] 三 seed 的 1024 条 rollout 结果已生成"
echo "[HINT] 随后运行：python tools/compare_with_unet_raw.py --n-samples 1024 \\"
echo "         --pred-subdir baseline_eval_rollout18_full"
