#!/bin/bash
#SBATCH --job-name=simvp_full
#SBATCH --output=/home/group1/26fall_aiclass/cb/cira-diff/reproduction/simvp/full_train/logs/output/simvp_full_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/cb/cira-diff/reproduction/simvp/full_train/logs/err/simvp_full_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#
# 提交方式（logs/output 与 logs/err 必须已存在，SLURM 不会自动创建父目录）：
#     cd /home/group1/26fall_aiclass/cb/cira-diff/reproduction/simvp/full_train
#     sbatch run_full_train.slurm
#
# 查看状态 / 日志：
#     squeue -u $USER
#     tail -f logs/output/simvp_full_<jobid>.out
#
# 任务：GOES-16 ABI ch13 全量训练 SimVP（2 帧输入 -> 1 帧输出，单步）。
#       train / validation / test 三份 zarr 各司其职，训练后在全 test 集上
#       评估 RMSE（亮温 K）并与 persistence 基线对比。
#
# 超参**全部**在 OpenSTL/configs/satcast/SimVP.py 里，本脚本不重复定义，
# 避免两处数值漂移。这里只放运行环境/路径/规模控制。

set -euo pipefail

# ---- 可按需修改 ----
CONDA_ENV=cira-diff-cb
WORKDIR=/home/group1/26fall_aiclass/cb/cira-diff/reproduction/simvp/full_train
OPENSTL_DIR="$WORKDIR/OpenSTL"
CONDA_BASE=/home/group1/miniconda3
DATA_ROOT=/data1/satcast
EX_NAME=satcast_simvp_full
EVAL_ZARR=edm_GOES_ch13_test_dataset.zarr

# 冒烟测试：改成 1 则只跑 3 条样本（train/val/test 各 1 条）2 个 epoch，
# 用于在提交长任务前确认三份数据通路 + 代码改动都通。产物写到独立目录，不会污染正式结果。
# 也可以不改文件、直接在命令行覆盖：sbatch --export=ALL,SMOKE=1 run_full_train.slurm
SMOKE=${SMOKE:-0}
# 续跑：改成 1 则若存在 checkpoints/last.ckpt 就从它接着训（改动超参后请保持 0）
RESUME=${RESUME:-0}
# --------------------

if [ "$SMOKE" = "1" ]; then
    RUN_NAME="${EX_NAME}_smoke"
    # SMOKE_EPOCHS 可覆盖：把第一次设成 2、第二次设成 4 配合 RESUME=1，
    # 就能真正验证"从 last.ckpt 接着训"这条路径。
    SMOKE_ARGS=(--single_sample_idx 0 -e "${SMOKE_EPOCHS:-2}")
else
    RUN_NAME="$EX_NAME"
    SMOKE_ARGS=()
fi

OUT_ROOT="$WORKDIR/outputs/$RUN_NAME"
mkdir -p "$WORKDIR/logs/output" "$WORKDIR/logs/err" "$WORKDIR/outputs"

cd "$OPENSTL_DIR"

# 激活 conda 环境
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# 校验：必须真的跑在 cira-diff-cb 里。本机 PATH 里可能残留别的同学的环境
# （比如 cira-diff-ly），一旦 python 解析到那个环境上，作业会"看起来成功、
# 实际跑在别人的环境里"。所以显式把本环境的 bin 放到 PATH 最前面并断言。
EXPECT_ENV="$CONDA_BASE/envs/$CONDA_ENV"
export PATH="$EXPECT_ENV/bin:$PATH"
if [ ! -x "$EXPECT_ENV/bin/python" ]; then
    echo "[ERROR] 找不到环境 $EXPECT_ENV，无法继续" >&2
    exit 1
fi
if [ "$(which python)" != "$EXPECT_ENV/bin/python" ]; then
    echo "[ERROR] python 解析到了 $(which python)，但期望是 $EXPECT_ENV/bin/python" >&2
    exit 1
fi

# OpenSTL 不是 pip 包（没有 setup develop），靠 PYTHONPATH 让它可被 import
export PYTHONPATH="$OPENSTL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "======================================================================"
echo "[INFO] which python : $(which python)"
echo "[INFO] python ver   : $(python -V 2>&1)"
echo "[INFO] conda env    : ${CONDA_DEFAULT_ENV:-N/A}"
echo "[INFO] torch ver    : $(python -c 'import torch; print(torch.__version__)')"
echo "[INFO] lightning    : $(python -c 'import lightning; print(lightning.__version__)')"
echo "[INFO] job id       : ${SLURM_JOB_ID:-N/A}"
echo "[INFO] job name     : ${SLURM_JOB_NAME:-N/A}"
echo "[INFO] node         : $(hostname)"
echo "[INFO] workdir      : $OPENSTL_DIR"
echo "[INFO] data         : $DATA_ROOT  (train/validation/test 三份 zarr)"
echo "[INFO] run name     : $RUN_NAME  (SMOKE=$SMOKE RESUME=$RESUME)"
echo "[INFO] gpu (CUDA_VISIBLE_DEVICES) : ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "[INFO] gpu name     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "[INFO] start time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

# 续跑：Lightning 的 fit(ckpt_path=...) 会恢复 epoch / optimizer / scheduler 状态
RESUME_ARGS=()
if [ "$RESUME" = "1" ] && [ -f "$OUT_ROOT/checkpoints/last.ckpt" ]; then
    echo "[INFO] 发现 $OUT_ROOT/checkpoints/last.ckpt，从该 checkpoint 续跑"
    RESUME_ARGS=(--ckpt_path "$OUT_ROOT/checkpoints/last.ckpt")
fi

# 超参见 configs/satcast/SimVP.py；--data_root 只是兜底（config 里也有，
# 且本副本的 tools/train.py 已把 data_root 放进 exclude_keys，配置为准）
set +e
python -u tools/train.py \
    -d satcast \
    -m SimVP \
    -c configs/satcast/SimVP.py \
    --data_root "$DATA_ROOT" \
    --res_dir "$WORKDIR/outputs" \
    --ex_name "$RUN_NAME" \
    "${SMOKE_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
STATUS=$?
set -e

# 训练结束后做推理评估：加载 best.ckpt，在完整 test 集上与 persistence 基线对比。
# 注意：OpenSTL 自己的 trainer.test() 默认跑在**最终权重**上（不是 best.ckpt），
# 所以这里用独立脚本重新加载 best.ckpt 评估，那个才是权威数字。
if [ "$STATUS" -eq 0 ]; then
    echo "======================================================================"
    echo "[INFO] 开始测试集评估（加载 best.ckpt + persistence 基线对比）"
    echo "======================================================================"
    set +e
    python -u "$WORKDIR/eval_full_test.py" \
        --ckpt "$OUT_ROOT/checkpoints/best.ckpt" \
        --config "$OPENSTL_DIR/configs/satcast/SimVP.py" \
        --data_root "$DATA_ROOT" \
        --zarr_name "$EVAL_ZARR" \
        --out_dir "$OUT_ROOT" \
        --batch_size 32
    EVAL_STATUS=$?
    set -e
    if [ "$EVAL_STATUS" -ne 0 ]; then
        echo "[WARNING] 评估脚本以非 0 状态退出（$EVAL_STATUS），训练本身是成功的"
    fi
fi

echo "======================================================================"
echo "[INFO] end time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] exit code    : $STATUS"
if [ "$STATUS" -eq 0 ]; then
    echo "[INFO] 训练脚本正常退出"
else
    echo "[ERROR] 训练脚本以非 0 状态退出，请查看本 .out 文件与同名的 .err 文件"
fi
echo "======================================================================"
exit "$STATUS"
