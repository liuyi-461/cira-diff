#!/bin/bash
#SBATCH --job-name=eval_rollout1024
#SBATCH --output=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/eval_rollout1024_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/eval_rollout1024_%j.err
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
