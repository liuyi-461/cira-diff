#!/bin/bash
#SBATCH --job-name=simvp_rollout
#SBATCH --output=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/train_rollout_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/train_rollout_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#
# 多步滚动训练 + scheduled sampling —— GAN 方案的**对照实验**。
# 不引入任何生成式损失，只用「多步 MSE」+「按概率喂自己的预测」。
#
# 训练数据用 validation 集（其 output 有 18 帧真值，可支撑多步监督）；
# **test 集始终 held-out**，评估口径与 baseline / GAN 一致。
#
# 环境变量：
#   SCHED=0.0  纯多步（teacher forcing）
#   SCHED=0.5  + scheduled sampling
#   ROLLOUT=6  滚动步数
#   EPOCHS=20  轮数（配合 800 样本/batch2 = 400 步/轮，20 轮 ≈ 8000 步，
#              与 GAN 诊断的梯度步数预算对齐，保证对照公平）
#
# ⚠️ 注意：本对照与 GAN 的**训练数据来源不同**（validation 800 条 vs train 35595 条），
#    对照的是"同样热启动、同样步数预算下，哪种改进方向更有效"，报告时需注明。

set -euo pipefail

CONDA_ENV=openstl
CONDA_BASE=/home/group1/miniconda3
OPENSTL_DIR=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL
LOGDIR="$OPENSTL_DIR/test_dl/logs"
DATA_ROOT=/home/group1/26fall_aiclass/yr/data1/data1/satcast
CONFIG="$OPENSTL_DIR/configs/satcast/SimVP.py"

# ----------------------------- 可调参数 -----------------------------
SCHED=${SCHED:-0.5}
ROLLOUT=${ROLLOUT:-6}
EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-2}
LR=${LR:-5e-4}
SEED=${SEED:-42}
N_TRAIN=${N_TRAIN:-800}
N_VAL=${N_VAL:-224}
# ex_name 带上 p，便于区分两个子臂
EX_NAME=${EX_NAME:-satcast_simvp_rollout_ss${SCHED}}
INIT_CKPT=${INIT_CKPT:-/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/work_dirs/satcast_simvp_full/checkpoints/best.ckpt}
# -------------------------------------------------------------------

mkdir -p "$LOGDIR"
cd "$OPENSTL_DIR"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$OPENSTL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "[INFO] job id : ${SLURM_JOB_ID:-N/A}"
echo "[INFO] node   : $(hostname)"
echo "[INFO] ex_name=$EX_NAME  rollout=$ROLLOUT  sched_samp=$SCHED"
echo "[INFO] epochs=$EPOCHS  batch=$BATCH  lr=$LR  seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true

[ -f "$CONFIG" ] || { echo "[FATAL] config 不存在: $CONFIG"; exit 3; }
python -u -c "import torch; print('[OK] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

ARGS=(--config "$CONFIG" --data-root "$DATA_ROOT" --res-dir "$OPENSTL_DIR/work_dirs"
      --ex-name "$EX_NAME" --rollout-steps "$ROLLOUT" --sched-samp "$SCHED"
      --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" --seed "$SEED"
      --n-train "$N_TRAIN" --n-val "$N_VAL")
if [ -n "$INIT_CKPT" ] && [ -f "$INIT_CKPT" ]; then
    ARGS+=(--init-ckpt "$INIT_CKPT")
    echo "[INFO] 热启动自: $INIT_CKPT"
fi

python -u "$OPENSTL_DIR/tools/train_simvp_rollout.py" "${ARGS[@]}"

echo "[DONE] 多步训练产物: $OPENSTL_DIR/work_dirs/$EX_NAME"
echo "[HINT] 随后用 tools/eval_simvp_baseline.py 在 test 集（1024 条 × 18 步）评估，"
echo "       与 baseline / GAN 的 RMSE 与锐度比对照。"
