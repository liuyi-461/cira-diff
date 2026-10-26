#!/bin/bash
#SBATCH --job-name=productA
#SBATCH --output=test_dl/logs/train_productA_%j.out
#SBATCH --error=test_dl/logs/train_productA_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#
# 产品 A：SimVP 主产品 + K=6 GAN（gan : le = 1 : 4），只用 train 集。
#
# 两路配置（同预算对照，隔离「GAN 在 1:4 权重下是帮忙还是帮倒忙」）：
#
#   CONFIG=gan    主配置   : gan_weight=0.25 -> loss = mse*(1 + 0.25*adv)
#                            即 对抗项 : MSE项 = 0.25 : 1 = 1 : 4   ★用户指定
#   CONFIG=nogan  同预算对照: gan_weight=0    -> loss = mse
#                            （D 仍更新但不影响 G，浪费少量算存，换来做干净对照）
#
# 共同点：
#   * K=6 滚动（覆盖前 1 小时，与"重点关注前 1h"一致）
#   * 只用 train 集 35595 条（output 仅 1 帧真值 -> MSE 只能监督第 1 步，
#     第 2~6 步靠 GAN 的分布匹配信号）
#   * 判别器真实样本 = 本批 3 帧真实观测；假样本 = K*B 帧滚动预报
#   * 热启动自 baseline best.ckpt（100 epoch 训好的单步模型）
#
# ⚠️ 不要设 CUDA_VISIBLE_DEVICES：GPU 由 SLURM 分配。

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-openstl}
CONDA_BASE=${CONDA_BASE:-/home/group1/miniconda3}
CONFIG=${CONFIG:-gan}

OPENSTL_DIR="${OPENSTL_DIR:-}"
if [ -z "$OPENSTL_DIR" ]; then
    for _cand in "${SLURM_SUBMIT_DIR:-}" "$(pwd)"; do
        if [ -n "$_cand" ] && [ -d "$_cand/tools" ] && [ -d "$_cand/openstl" ]; then
            OPENSTL_DIR="$_cand"; break
        fi
    done
fi
OPENSTL_DIR="${OPENSTL_DIR:-/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL}"
if [ ! -d "$OPENSTL_DIR/tools" ]; then
    echo "[FATAL] 仓库根解析失败：$OPENSTL_DIR" >&2; exit 3
fi

LOGDIR="$OPENSTL_DIR/test_dl/logs"
mkdir -p "$LOGDIR"
cd "$OPENSTL_DIR"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$OPENSTL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8

# ----------------------------- 可调参数 -----------------------------
K=${K:-6}
EPOCHS=${EPOCHS:-2}
BATCH=${BATCH:-2}
LR=${LR:-2e-3}
GAN_W=${GAN_W:-}          # 留空则按 CONFIG 取默认；CONFIG=cont 时必填
GRAD_W=${GRAD_W:-}
INIT_OVERRIDE=${INIT_OVERRIDE:-}   # 留空则用 INIT_CKPT
INIT_CKPT=${INIT_CKPT:-$OPENSTL_DIR/work_dirs/satcast_simvp_full/checkpoints/best.ckpt}
# --------------------------------------------------------------------

case "$CONFIG" in
  gan)   EX_NAME=productA_gan_k${K};   GW=${GAN_W:-0.25}; GRW=${GRAD_W:-0} ;;
  nogan) EX_NAME=productA_nogan_k${K}; GW=${GAN_W:-0};    GRW=${GRAD_W:-0} ;;
  # cont：从旧 simvp+gan 的 ckpt 续训（保留其 0.05 + 双向锐度正则配置），
  #       数据换成全量 train、LR 下调（续训不应再用峰值学习率）。
  cont)  EX_NAME=${EX_NAME:-productA_cont_k${K}}
         GW=${GAN_W:-0.05}; GRW=${GRAD_W:-0.05}
         INIT_CKPT=${INIT_OVERRIDE:-$OPENSTL_DIR/work_dirs/satcast_simvp_gan/checkpoints/best.ckpt} ;;
  *) echo "[FATAL] 未知 CONFIG=$CONFIG（gan / nogan / cont）" >&2; exit 3 ;;
esac

echo "[INFO] job=${SLURM_JOB_ID:-N/A} node=$(hostname)"
echo "[INFO] CONFIG=$CONFIG  ex_name=$EX_NAME"
echo "[INFO] K=$K  epochs=$EPOCHS  batch=$BATCH  gan_weight=$GW  (grad_weight=0)"
echo "[INFO] gan:le = $GW : 1"
echo "[INFO] 数据: train 35595 条（只用 train，不用 validation）"
echo "[INFO] 热启动: $INIT_CKPT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | head -1 || true
[ -f "$INIT_CKPT" ] || { echo "[FATAL] 缺热启动权重：$INIT_CKPT"; exit 3; }

python -u "$OPENSTL_DIR/tools/train_simvp_gan.py" \
    --config "$OPENSTL_DIR/configs/satcast/SimVP.py" \
    --init-ckpt "$INIT_CKPT" \
    --rollout-steps "$K" \
    --gan-weight "$GW" \
    --grad-weight "$GRW" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --num-workers 8 \
    --ex-name "$EX_NAME"

echo "[DONE] 产品 A 产物: $OPENSTL_DIR/work_dirs/$EX_NAME"
echo "[HINT] 出 18 步 test 预测:"
echo "  python tools/eval_simvp_baseline.py --ckpt .../$EX_NAME/checkpoints/best.ckpt \\"
echo "      --config configs/satcast/SimVP.py --norm-mean 0 --norm-std 1 \\"
echo "      --rollout-steps 18 --out_dir .../$EX_NAME/eval_rollout18"
