#!/bin/bash
#SBATCH --job-name=eval_seeds
#SBATCH --output=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/eval_seeds_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/eval_seeds_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# 两个补充 seed 训练完成后：统一协议评估（test 1024 条 × 18 步）+ 导出 npy
# + 五臂对比（含全量 SS 臂与其 2 个 seed）+ 计算 3-seed 的 mean±std。
#
# 由 launch_seeds.slurm 自动提交（依赖两个训练作业）。

set -euo pipefail

CONDA_ENV=openstl
CONDA_BASE=/home/group1/miniconda3
OPENSTL_DIR=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL
LOGDIR="$OPENSTL_DIR/test_dl/logs"
DATA_ROOT=/home/group1/26fall_aiclass/yr/data1/data1/satcast
CONFIG="$OPENSTL_DIR/configs/satcast/SimVP.py"

mkdir -p "$LOGDIR"; cd "$OPENSTL_DIR"
source "$CONDA_BASE/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"
export PYTHONPATH="$OPENSTL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"
mkdir -p "$MPLCONFIGDIR"

echo "[INFO] job id : ${SLURM_JOB_ID:-N/A}"

for D in satcast_simvp_rollout_ss05_full_seed1 satcast_simvp_rollout_ss05_full_seed7; do
    CKPT="$OPENSTL_DIR/work_dirs/$D/checkpoints/best.ckpt"
    if [ ! -f "$CKPT" ]; then
        echo "[WARN] 跳过 $D（无 best.ckpt）"
        continue
    fi
    echo "================= 评估 $D（1024 条 × 18 步） ================="
    python -u "$OPENSTL_DIR/tools/eval_simvp_baseline.py" \
        --ckpt "$CKPT" --config "$CONFIG" --data_root "$DATA_ROOT" \
        --out_dir "$OPENSTL_DIR/work_dirs/$D/baseline_eval_rollout18_full" \
        --rollout-steps 18 --max-samples 1024 --batch_size 16 \
        --norm-mean 0 --norm-std 1
done

echo "================= 导出 npy（官方格式） ================="
python -u "$OPENSTL_DIR/tools/export_preds_official_format.py" \
    --extra rollout_ss0.0=satcast_simvp_rollout_ss0.0 \
            rollout_ss0.5=satcast_simvp_rollout_ss0.5 \
            rollout_ss05_full=satcast_simvp_rollout_ss05_full \
            rollout_ss05_full_seed1=satcast_simvp_rollout_ss05_full_seed1 \
            rollout_ss05_full_seed7=satcast_simvp_rollout_ss05_full_seed7 \
            gan=satcast_simvp_gan

echo "================= 五臂对比 + 3-seed 统计 ================="
python -u "$OPENSTL_DIR/tools/compare_arms.py"

echo "[DONE] 补充 seed 评估完成，见 docs/experiments/EXP-006-anti-blur/"
