#!/bin/bash
#SBATCH --job-name=simvp_finalize
#SBATCH --output=test_dl/logs/finalize_baseline_%j.out
#SBATCH --error=test_dl/logs/finalize_baseline_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --dependency=afterany:213
#
# 训练(213) 结束后自动完成"重测 + 回填正式结果"：
#   1) 用最终 best.ckpt 重跑单步评估（test 全量 1024 条）
#   2) 重跑 18 步 rollout（128 条子集，对齐 ly）
#   3) 调用 finalize_baseline_report.py 自动生成 EXP-004 的 Results 段
#
# 注意：--dependency 写死了 213。若后续又续跑出新作业号，需改这里的作业号，
#       或在提交时用 sbatch --dependency=afterany:<新jobid> test_dl/finalize_baseline.slurm 覆盖。
#
# 说明：训练脚本自身在成功结束时也会跑一次评估；这里再跑一遍是为了兜底——
#       万一训练被信号中断（如 205 那样）没能走到评估步骤，本作业仍会产出最终数字。

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
RUN_DIR="$OPENSTL_DIR/work_dirs/satcast_simvp_full"
BEST_CKPT="$RUN_DIR/checkpoints/best.ckpt"

mkdir -p "$LOGDIR"
cd "$OPENSTL_DIR"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$OPENSTL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"
mkdir -p "$MPLCONFIGDIR"

echo "[INFO] job id  : ${SLURM_JOB_ID:-N/A}"
echo "[INFO] ckpt    : $BEST_CKPT"

if [ ! -f "$BEST_CKPT" ]; then
    echo "[FATAL] 找不到 $BEST_CKPT，无法评估"
    exit 3
fi

python -u -c "
import torch
ck=torch.load('$BEST_CKPT',map_location='cpu',weights_only=False)
print('[INFO] 最终 best.ckpt 的 epoch =', ck.get('epoch','N/A'))
"

echo "====================== 1) 单步评估（1024 条） ======================"
python -u "$OPENSTL_DIR/tools/eval_simvp_baseline.py" \
    --ckpt "$BEST_CKPT" --config "$CONFIG" --data_root "$DATA_ROOT" \
    --out_dir "$RUN_DIR/baseline_eval" --rollout-steps 1 \
    --norm-mean 0 --norm-std 1

echo "====================== 2) 18 步 rollout 评估 ======================"
python -u "$OPENSTL_DIR/tools/eval_simvp_baseline.py" \
    --ckpt "$BEST_CKPT" --config "$CONFIG" --data_root "$DATA_ROOT" \
    --out_dir "$RUN_DIR/baseline_eval_rollout18" --rollout-steps 18 \
    --norm-mean 0 --norm-std 1

echo "====================== 3) 生成正式 Results 段 ======================"
python -u "$OPENSTL_DIR/tools/finalize_baseline_report.py" \
    --single "$RUN_DIR/baseline_eval/baseline_metrics.json" \
    --rollout "$RUN_DIR/baseline_eval_rollout18/baseline_metrics.json"

echo "[DONE] 正式结果已写入 docs/experiments/EXP-004-simvp-vs-baseline/README.md"
