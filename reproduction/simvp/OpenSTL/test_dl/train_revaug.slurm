#!/bin/bash
#SBATCH --job-name=revaug
#SBATCH --output=test_dl/logs/train_revaug_%j.out
#SBATCH --error=test_dl/logs/train_revaug_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#
# 反向任务增广训练（forecast-realism-report.md §10 方案 1）
#
# 三路配置（梯度步数严格对齐，隔离「反向任务」单因素）：
#
#   CONFIG=full    全量处理   : 35595 -> 71190 条, 50 epoch  = 444,900 步
#                             （与 baseline 100ep x 4449 步严格对齐）
#   CONFIG=q       1/4 处理   :  8899 -> 17798 条, 50 epoch  = 111,200 步
#   CONFIG=qctrl   1/4 对照   :  8899          条, 100 epoch = 111,200 步
#                             （与 CONFIG=q 严格对齐）
#
# 验收线（§10.5 预注册）：单步 LT=1 RMSE 相对 baseline(1.5761 K) 改善 >=2%
#   -> 需 <= 1.545 K
#
# ⚠️ 不要设 CUDA_VISIBLE_DEVICES：GPU 由 SLURM 分配，设了会覆盖调度结果。

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-openstl}
CONDA_BASE=${CONDA_BASE:-/home/group1/miniconda3}
CONFIG=${CONFIG:-full}

# 可移植性：仓库根按优先级取 OPENSTL_DIR > SLURM_SUBMIT_DIR（校验含 tools/ 与 openstl/）> 内置默认值。
# ⚠️ 不能用 ${BASH_SOURCE[0]} 推导：SLURM 会把脚本复制到 /var/spool/slurmd/jobNNNN/slurm_script
#    再执行，BASH_SOURCE 指向 spool 而非原路径（实测踩过坑）。
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
    echo "[FATAL] 请从 OpenSTL 根目录执行 sbatch test_dl/train_revaug.slurm，或显式 export OPENSTL_DIR=<仓库根>" >&2
    exit 3
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

# ----------------------------- 三路配置 -----------------------------
case "$CONFIG" in
  full)
    EX_NAME=revaug_full_e50
    EPOCHS=50
    LIMIT=""                       # 全量，不限
    REVAUG="--reverse-aug"
    ;;
  q)
    EX_NAME=revaug_q_e50
    EPOCHS=50
    LIMIT="--limit-train 8899"
    REVAUG="--reverse-aug"
    ;;
  qctrl)
    EX_NAME=revaug_q_ctrl_e100
    EPOCHS=100
    LIMIT="--limit-train 8899"
    REVAUG=""                      # 对照：不增广
    ;;
  *)
    echo "[FATAL] 未知 CONFIG=$CONFIG（可选 full / q / qctrl）" >&2
    exit 3
    ;;
esac
# --------------------------------------------------------------------

echo "[INFO] job id  : ${SLURM_JOB_ID:-N/A}"
echo "[INFO] node    : $(hostname)"
echo "[INFO] OPENSTL : $OPENSTL_DIR"
echo "[INFO] CONFIG  : $CONFIG  ex_name=$EX_NAME"
echo "[INFO] epochs=$EPOCHS  limit='${LIMIT:-全量}'  reverse_aug='${REVAUG:-无}'"
echo "[INFO] GPU     : $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | head -1 || true

python -u "$OPENSTL_DIR/tools/train_simvp_satcast.py" \
    --config "$OPENSTL_DIR/configs/satcast/SimVP.py" \
    --epochs "$EPOCHS" \
    --ex_name "$EX_NAME" \
    $LIMIT \
    $REVAUG

echo "[DONE] 产物: $OPENSTL_DIR/work_dirs/$EX_NAME"
echo "[HINT] 评估（与 baseline 同协议）:"
echo "  python tools/eval_simvp_baseline.py \\"
echo "      --ckpt $OPENSTL_DIR/work_dirs/$EX_NAME/checkpoints/best.ckpt \\"
echo "      --norm-mean 0 --norm-std 1"
echo "[HINT] 对照锚点: baseline 单步 LT=1 RMSE = 1.5761 K（需 <= 1.545 K 才算通过）"
