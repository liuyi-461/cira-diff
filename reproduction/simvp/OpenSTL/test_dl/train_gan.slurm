#!/bin/bash
#SBATCH --job-name=simvp_gan
#SBATCH --output=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/train_gan_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/train_gan_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#
# SimVP + GAN loss 的滚动预报训练（抗长时效模糊）。
#
# 核心：训练时滚动 K 步，把滚动结果送判别器；判别器不需要多步真值
#       （train 集只有 1 帧真值），只判断"像不像真实云图"。
#
# 环境变量覆盖示例：
#   SMOKE=1 sbatch test_dl/train_gan.slurm              # 冒烟：少量样本 2 epoch
#   ROLLOUT=4 GAN_WEIGHT=0.02 sbatch test_dl/train_gan.slurm
#   INIT_CKPT=<baseline best.ckpt> sbatch ...            # 从 baseline 热启动（推荐）
#
# ⚠️ 显存：滚动 K 步 = K 倍计算图。默认 batch=4 / K=3。
#    OOM 时：减小 BATCH，或加 DETACH=1（只对最后一步回传梯度）。

set -euo pipefail

CONDA_ENV=openstl
CONDA_BASE=/home/group1/miniconda3
OPENSTL_DIR=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL
LOGDIR="$OPENSTL_DIR/test_dl/logs"
DATA_ROOT=/home/group1/26fall_aiclass/yr/data1/data1/satcast
CONFIG="$OPENSTL_DIR/configs/satcast/SimVP.py"

# ----------------------------- 可调参数 -----------------------------
EX_NAME=${EX_NAME:-satcast_simvp_gan}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-4}
ROLLOUT=${ROLLOUT:-3}                  # 训练时滚动几步
GAN_WEIGHT=${GAN_WEIGHT:-0.01}         # 对抗损失权重
GRAD_WEIGHT=${GRAD_WEIGHT:-0.0}        # 锐度正则（0=关闭）
LR=${LR:-2e-3}
LR_D=${LR_D:-1e-4}
SEED=${SEED:-42}
SMOKE=${SMOKE:-0}
DETACH=${DETACH:-0}
# 诊断模式：只取前 N 条训练样本（如 8000），用于快速验证超参是否会把模型带偏
LIMIT_TRAIN=${LIMIT_TRAIN:-}
# 推荐：从 baseline 的 best.ckpt 热启动，避免从头学
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

echo "[INFO] job id   : ${SLURM_JOB_ID:-N/A}"
echo "[INFO] node     : $(hostname)"
echo "[INFO] ex_name  : $EX_NAME"
echo "[INFO] rollout=$ROLLOUT  gan_weight=$GAN_WEIGHT  grad_weight=$GRAD_WEIGHT"
echo "[INFO] batch=$BATCH  epochs=$EPOCHS  lr_G=$LR  lr_D=$LR_D  seed=$SEED"
# 注意：nvidia-smi 输出多行时 `| head -1` 会让 nvidia-smi 收到 SIGPIPE，
# 在 set -o pipefail 下会直接终止脚本（曾导致作业 0 秒失败）。故加 `|| true`。
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true

[ -f "$CONFIG" ] || { echo "[FATAL] config 不存在: $CONFIG"; exit 3; }
python -u -c "import torch; print('[OK] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

ARGS=(--config "$CONFIG" --data-root "$DATA_ROOT" --res-dir "$OPENSTL_DIR/work_dirs"
      --ex-name "$EX_NAME" --epochs "$EPOCHS" --batch-size "$BATCH"
      --rollout-steps "$ROLLOUT" --gan-weight "$GAN_WEIGHT"
      --grad-weight "$GRAD_WEIGHT" --lr "$LR" --lr-d "$LR_D" --seed "$SEED")

if [ "$SMOKE" = "1" ]; then
    ARGS+=(--epochs 2 --limit-train 64 --limit-val 16 --limit-test 16 --log-every 5)
    echo "[INFO] *** 冒烟模式 ***"
fi
if [ "$DETACH" = "1" ]; then
    ARGS+=(--detach-steps)
fi
if [ -n "$LIMIT_TRAIN" ]; then
    ARGS+=(--limit-train "$LIMIT_TRAIN")
    echo "[INFO] 诊断模式：只取前 $LIMIT_TRAIN 条训练样本"
fi
if [ -n "$INIT_CKPT" ] && [ -f "$INIT_CKPT" ]; then
    ARGS+=(--init-ckpt "$INIT_CKPT")
    echo "[INFO] 热启动自: $INIT_CKPT"
else
    echo "[INFO] 从头训练（未热启动）"
fi

python -u "$OPENSTL_DIR/tools/train_simvp_gan.py" "${ARGS[@]}"

echo "[DONE] GAN 训练产物: $OPENSTL_DIR/work_dirs/$EX_NAME"
echo "[HINT] 关注 gan_history.json 里的 sharpness_ratio（pred 梯度/real 梯度），"
echo "       越接近 1 越不模糊；baseline 通常明显 < 1。"
