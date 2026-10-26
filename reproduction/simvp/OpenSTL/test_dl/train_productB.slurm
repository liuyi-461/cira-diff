#!/bin/bash
#SBATCH --job-name=productB
#SBATCH --output=test_dl/logs/train_productB_%j.out
#SBATCH --error=test_dl/logs/train_productB_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#
# 产品 B：独立残差预报产品（修正产品 A 的滚动预报）。
#
# 两路配置（同预算对照，隔离「B 的 GAN 在 1:4 下是帮忙还是帮倒忙」）：
#   CONFIG=gan    主配置 : gan_weight=0.2 -> gan:le = 1:5
#                          （依据 tools/probe_gan_weight.py 权重探针：锐度增益在
#                            w≈0.1 后饱和而 RMSE 代价仍升，0.2 取增益显著段）
#   CONFIG=nogan  对照   : gan_weight=0    -> 纯 MSE 拟合残差
# 两路都用**同一个**产品 A（--a-ckpt 指向 productA_gan_k6），
# 这样唯一变量是 B 的 GAN 权重。
#
# 关键设计（与旧 train_residual*.py 的区别）：
#   * A 独立自回归，反馈只用 â；B 不进入 A 的反馈环 -> 两产品解耦
#   * B 是**逐步修正器**：输入 [f_t-10, f_t, â_k, lead_k] -> 残差场
#   * 最终产品 = â_k + B_k，只在末端相加
#   * B 输出层零初始化 -> 初始 B≡0，最终产品严格等于 A
#
# ⚠️ 数据约束：train 的 output 只有 1 帧真值，残差目标 r_k 只有 k=1 可得，
#   故 B 的 MSE 只能监督第 1 步；第 2~6 步靠 GAN 的分布匹配信号。
#   后果：B 的修正从第 1 步外推到其它步，收益预期集中在短时效。
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
BATCH=${BATCH:-16}
A_CKPT=${A_CKPT:-$OPENSTL_DIR/work_dirs/productA_gan_k6/checkpoints/best.ckpt}
# --------------------------------------------------------------------

# EX_NAME 可用环境变量覆盖；默认按「挂在哪个 A 上 + 配置」派生，
# 避免不同 A 的 B 写到同一目录互相覆盖（曾因 EX_NAME 撞车覆盖过产物）。
A_TAG=${A_TAG:-ongan}          # ongan=挂在 A-gan 上 / onbase=挂在 baseline 上
case "$CONFIG" in
  gan)   EX_NAME=${EX_NAME:-productB_${A_TAG}_gan_k${K}};   GW=0.2 ;;
  nogan) EX_NAME=${EX_NAME:-productB_${A_TAG}_nogan_k${K}}; GW=0 ;;
  *) echo "[FATAL] 未知 CONFIG=$CONFIG（gan / nogan）" >&2; exit 3 ;;
esac

echo "[INFO] job=${SLURM_JOB_ID:-N/A} node=$(hostname)"
echo "[INFO] CONFIG=$CONFIG  ex_name=$EX_NAME  gan_weight=$GW (gan:le = $GW:1)"
echo "[INFO] K=$K epochs=$EPOCHS batch=$BATCH"
echo "[INFO] 产品 A: $A_CKPT"
echo "[INFO] 数据: 只用 train 35595 条"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | head -1 || true
[ -f "$A_CKPT" ] || { echo "[FATAL] 缺产品 A 权重：$A_CKPT"; exit 3; }

python -u "$OPENSTL_DIR/tools/train_residual_product.py" \
    --config "$OPENSTL_DIR/configs/satcast/SimVP.py" \
    --a-ckpt "$A_CKPT" \
    --ex-name "$EX_NAME" \
    --rollout-steps "$K" \
    --gan-weight "$GW" \
    --width 48 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --num-workers 8 \
    --log-every 200

echo "[DONE] 产品 B 产物: $OPENSTL_DIR/work_dirs/residual/$EX_NAME"
echo "[HINT] 出最终产品 A+B 的 18 步预测:"
echo "  python tools/make_final_preds.py --ex-name $EX_NAME \\"
echo "      --a-ckpt $A_CKPT --out work_dirs/residual/$EX_NAME/final_preds.npy"
echo "[HINT] 再评估:"
echo "  python tools/eval_metrics.py --preds <上一步输出> \\"
echo "      --out-dir docs/experiments/EXP-008-metrics-baseline/$EX_NAME/ \\"
echo "      --leads 1,2,3,6,9,12,15,18"
