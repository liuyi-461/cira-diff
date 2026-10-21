#!/bin/bash
#SBATCH --job-name=gan_ss
#SBATCH --output=test_dl/logs/train_gan_ss_%j.out
#SBATCH --error=test_dl/logs/train_gan_ss_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#
# GAN + scheduled sampling 联合训练（对抗损失 × 曝光偏差修正）。
#
# 为什么用 validation zarr：scheduled sampling 与多步 MSE 都**必须有多帧真值**，
# 只有 validation 集满足（output 有 18 帧）；train 集只有 1 帧，SS 在数学上不成立。
#
# 默认配置与既有「SS 全量 1024」臂**完全一致**，仅额外叠加对抗损失
# → 构成干净的 2×2 消融：baseline / GAN / SS / GAN+SS
#
# 环境变量覆盖示例：
#   SMOKE=1 sbatch test_dl/train_gan_ss.slurm            # 冒烟：少量样本 1 轮
#   EPOCHS=16 GAN_WEIGHT=0.05 GRAD_WEIGHT=0.05 sbatch ...   # 正式
#   SCHED=0.0 sbatch ...                                 # 关掉 SS（退化为纯多步+GAN）
#
# ⚠️ 显存：K=6 / batch=2 ≈ 29~32 GB，RTX 4090 (48GB) 安全。

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-openstl}
CONDA_BASE=${CONDA_BASE:-/home/group1/miniconda3}
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
if [ ! -d "$OPENSTL_DIR/tools" ] || [ ! -f "$OPENSTL_DIR/configs/satcast/SimVP.py" ]; then
    echo "[FATAL] 仓库根解析失败：$OPENSTL_DIR" >&2
    echo "[FATAL] 请从 OpenSTL 根目录执行 sbatch test_dl/train_gan_ss.slurm，或显式 export OPENSTL_DIR=<仓库根>" >&2
    exit 3
fi
LOGDIR="$OPENSTL_DIR/test_dl/logs"
# 可移植性：数据根优先取环境变量 SATCAST_ROOT
DATA_ROOT="${SATCAST_ROOT:-/home/group1/26fall_aiclass/yr/data1/data1/satcast}"
CONFIG="$OPENSTL_DIR/configs/satcast/SimVP.py"

# ----------------------------- 可调参数 -----------------------------
EX_NAME=${EX_NAME:-satcast_simvp_gan_ss}
ROLLOUT=${ROLLOUT:-6}              # 训练时滚动几步送入判别器
SCHED=${SCHED:-0.5}                # scheduled sampling 概率 p
GAN_WEIGHT=${GAN_WEIGHT:-0.05}     # 对抗损失（相对多步 mse 的比例）
GRAD_WEIGHT=${GRAD_WEIGHT:-0.05}   # 双向锐度正则（相对比例）
EPOCHS=${EPOCHS:-16}
BATCH=${BATCH:-2}
LR=${LR:-5e-4}
LR_D=${LR_D:-1e-4}
SEED=${SEED:-42}
N_TRAIN=${N_TRAIN:-1024}
N_VAL=${N_VAL:-512}
VAL_SOURCE=${VAL_SOURCE:-train}
SMOKE=${SMOKE:-0}
# 热启动：与 GAN / SS 臂用同一个 baseline best.ckpt，保证同起点可比
INIT_CKPT=${INIT_CKPT:-$OPENSTL_DIR/work_dirs/satcast_simvp_full/checkpoints/best.ckpt}
# -------------------------------------------------------------------

mkdir -p "$LOGDIR"
cd "$OPENSTL_DIR"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$OPENSTL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "[INFO] job id  : ${SLURM_JOB_ID:-N/A}"
echo "[INFO] node    : $(hostname)"
echo "[INFO] OPENSTL : $OPENSTL_DIR"
echo "[INFO] ex_name : $EX_NAME"
echo "[INFO] rollout=$ROLLOUT  sched_samp=$SCHED  gan_w=$GAN_WEIGHT  grad_w=$GRAD_WEIGHT"
echo "[INFO] epochs=$EPOCHS batch=$BATCH lr_G=$LR lr_D=$LR_D seed=$SEED"
echo "[INFO] data: n_train=$N_TRAIN n_val=$N_VAL val_source=$VAL_SOURCE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true

[ -f "$CONFIG" ] || { echo "[FATAL] config 不存在: $CONFIG"; exit 3; }
[ -d "$DATA_ROOT" ] || { echo "[FATAL] 数据根不存在: $DATA_ROOT"; exit 3; }
python -u -c "import torch; print('[OK] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

ARGS=(--config "$CONFIG" --data-root "$DATA_ROOT"
      --res-dir "$OPENSTL_DIR/work_dirs" --ex-name "$EX_NAME"
      --rollout-steps "$ROLLOUT" --sched-samp "$SCHED"
      --gan-weight "$GAN_WEIGHT" --grad-weight "$GRAD_WEIGHT"
      --epochs "$EPOCHS" --batch-size "$BATCH"
      --lr "$LR" --lr-d "$LR_D" --seed "$SEED"
      --n-train "$N_TRAIN" --n-val "$N_VAL" --val-source "$VAL_SOURCE")

if [ "$SMOKE" = "1" ]; then
    ARGS+=(--epochs 1 --n-train 64 --n-val 32 --log-every 5)
    echo "[INFO] *** 冒烟模式：64 训练 / 32 验证 / 1 轮 ***"
fi
if [ -n "$INIT_CKPT" ] && [ -f "$INIT_CKPT" ]; then
    ARGS+=(--init-ckpt "$INIT_CKPT")
    echo "[INFO] 热启动自: $INIT_CKPT"
else
    echo "[WARN] 未找到热启动权重，将从随机初始化开始训（不推荐）"
fi

python -u "$OPENSTL_DIR/tools/train_simvp_gan_ss.py" "${ARGS[@]}"

echo "[DONE] GAN+SS 训练产物: $OPENSTL_DIR/work_dirs/$EX_NAME"
echo "[HINT] 下一步评估（统一协议 test 1024 x 18 步 + 导出 npy）："
echo "       sbatch --dependency=afterany:${SLURM_JOB_ID:-<jobid>} test_dl/eval_gan_ss.slurm"
