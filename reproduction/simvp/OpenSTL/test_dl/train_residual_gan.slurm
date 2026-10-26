#!/bin/bash
#SBATCH --job-name=residual_gan
#SBATCH --output=test_dl/logs/train_residual_gan_%j.out
#SBATCH --error=test_dl/logs/train_residual_gan_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#
# SimVP + 残差分支 + 判别器 联合训练（残差与 GAN、SimVP 一起做云图预报）
#
# 三个消融臂（预注册，必须含 sham）：
#   R0      输入 [x̂, f_t]              —— 隔离残差分支本身的净贡献
#   R1      输入 [x̂, base, f_t]        —— 完整方案（含光流平流基）
#   R1sham  输入 [x̂, 0, f_t]           —— 隔离「平流基内容」vs「多一个输入通道」
#
# 判别器只判**残差场 Δ**（真实残差 = y−x̂，假残差 = R 输出），不判整场：
#   依据见 tools/train_residual_gan.py 顶部 docstring（LAPGAN 先例 +
#   本项目 GAN-only 压在整场导致锐度比 2.969 的反面实证）。
#
# 环境变量：
#   MODE=R0|R1|R1sham   EX_NAME=resgan_$MODE
#   SMOKE=1             32 训练 / 16 验证 / 1 轮 / K=3，测通路
#   EPOCHS=12  ROLLOUT=6  ALPHA=0.75  BATCH=8
#   GAN_WEIGHT=0.05  GRAD_WEIGHT=0.05  LR=1e-3  LR_D=1e-4
#   UNFREEZE=0         1 = S1 联合微调 SimVP 主干
#
# ⚠️ 为什么用 validation 集：train zarr 的 output 只有 1 帧真值，
#    逐步监督在数学上不成立；只有 validation 集有 18 帧真值。
# ⚠️ 本脚本不使用 scheduled sampling，也不与任何 SS 实验对比。

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

if [ ! -d "$OPENSTL_DIR/tools" ] || [ ! -f "$OPENSTL_DIR/configs/satcast/SimVP.py" ]; then
    echo "[FATAL] 仓库根解析失败：$OPENSTL_DIR" >&2
    echo "[FATAL] 缺 tools/ 或 configs/satcast/SimVP.py。" >&2
    echo "[FATAL] 请从 OpenSTL 根目录执行 sbatch test_dl/train_residual_gan.slurm，或显式 export OPENSTL_DIR=<仓库根>" >&2
    exit 3
fi

LOGDIR="$OPENSTL_DIR/test_dl/logs"
# 可移植性：数据根优先取环境变量 SATCAST_ROOT
DATA_ROOT="${SATCAST_ROOT:-/home/group1/26fall_aiclass/yr/data1/data1/satcast}"

# ----------------------------- 可调参数 -----------------------------
MODE=${MODE:-R1}                       # R0 / R1 / R1sham
TRAIN_ZARR=${TRAIN_ZARR:-edm_GOES_ch13_validation_dataset.zarr}
EX_NAME=${EX_NAME:-resgan_${MODE}}
ROLLOUT=${ROLLOUT:-6}
ALPHA=${ALPHA:-0.75}
EPOCHS=${EPOCHS:-12}
BATCH=${BATCH:-8}
LR=${LR:-1e-3}
LR_D=${LR_D:-1e-4}
GAN_WEIGHT=${GAN_WEIGHT:-0.05}
GRAD_WEIGHT=${GRAD_WEIGHT:-0.05}
WIDTH=${WIDTH:-32}
N_TRAIN=${N_TRAIN:-1024}
N_VAL=${N_VAL:-512}
VAL_SOURCE=${VAL_SOURCE:-train}
SEED=${SEED:-42}
SMOKE=${SMOKE:-0}
UNFREEZE=${UNFREEZE:-0}                # 1 = S1 联合微调 SimVP 主干
INIT_CKPT=${INIT_CKPT:-$OPENSTL_DIR/work_dirs/satcast_simvp_full/checkpoints/best.ckpt}
# --------------------------------------------------------------------

mkdir -p "$LOGDIR"
cd "$OPENSTL_DIR"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$OPENSTL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8

echo "[INFO] job id  : ${SLURM_JOB_ID:-N/A}"
echo "[INFO] node    : $(hostname)"
echo "[INFO] OPENSTL : $OPENSTL_DIR"
echo "[INFO] ex_name : $EX_NAME"
echo "[INFO] mode    : $MODE  (R0=无平流基 / R1=含平流基 / R1sham=平流基填中性值)"
echo "[INFO] 判别器作用于残差场 Δ（真实残差=y−x̂）"
echo "[INFO] rollout=$ROLLOUT alpha=$ALPHA epochs=$EPOCHS batch=$BATCH"
echo "[INFO] gan_weight=$GAN_WEIGHT grad_weight=$GRAD_WEIGHT lr=$LR lr_d=$LR_D"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | head -1 || true

[ -f "$INIT_CKPT" ] || { echo "[FATAL] 缺主模型权重：$INIT_CKPT"; exit 3; }
[ -d "$DATA_ROOT" ] || { echo "[FATAL] 数据根不存在：$DATA_ROOT"; exit 3; }
python -u -c "import cv2, torch; print('[OK] cv2', cv2.__version__, '| torch', torch.__version__)"

ARGS=(--config "$OPENSTL_DIR/configs/satcast/SimVP.py"
      --train-zarr "$TRAIN_ZARR"
      --data-root "$DATA_ROOT"
      --res-dir "$OPENSTL_DIR/work_dirs/residual"
      --ex-name "$EX_NAME"
      --mode "$MODE"
      --rollout-steps "$ROLLOUT"
      --alpha "$ALPHA"
      --epochs "$EPOCHS"
      --batch-size "$BATCH"
      --lr "$LR"
      --lr-d "$LR_D"
      --gan-weight "$GAN_WEIGHT"
      --grad-weight "$GRAD_WEIGHT"
      --width "$WIDTH"
      --n-train "$N_TRAIN"
      --n-val "$N_VAL"
      --val-source "$VAL_SOURCE"
      --seed "$SEED"
      --main-ckpt "$INIT_CKPT")

if [ "$SMOKE" = "1" ]; then
    ARGS+=(--epochs 1 --n-train 32 --n-val 16 --rollout-steps 3 --batch-size 4)
    echo "[INFO] *** 冒烟模式：32 训练 / 16 验证 / 1 轮 / K=3 ***"
fi
if [ "$UNFREEZE" = "1" ]; then
    ARGS+=(--unfreeze-main)
    echo "[INFO] S1 模式：SimVP 主干参与联合微调"
fi

python -u "$OPENSTL_DIR/tools/train_residual_gan.py" "${ARGS[@]}"

echo "[DONE] 产物: $OPENSTL_DIR/work_dirs/residual/$EX_NAME"
echo "[HINT] 下一步："
echo "  python tools/make_residual_preds.py --ex-name $EX_NAME \\"
echo "      --ckpt $OPENSTL_DIR/work_dirs/residual/$EX_NAME/checkpoints/best.pt"
echo "  python tools/eval_metrics.py --preds <上一步输出> \\"
echo "      --out-dir docs/experiments/EXP-008-metrics-baseline/$EX_NAME/"
