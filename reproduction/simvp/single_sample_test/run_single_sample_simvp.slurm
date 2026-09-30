#!/bin/bash
#SBATCH --job-name=simvp_single_sample
#SBATCH --output=/home/group1/26fall_aiclass/cb/cira-diff/reproduction/simvp/single_sample_test/logs/output/simvp_single_sample_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/cb/cira-diff/reproduction/simvp/single_sample_test/logs/err/simvp_single_sample_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# 提交方式（logs/output 与 logs/err 必须已存在，SLURM 不会自动创建父目录）：
#     cd /home/group1/26fall_aiclass/cb/cira-diff/reproduction/simvp/single_sample_test
#     sbatch run_single_sample_simvp.slurm
#
# 查看状态 / 日志：
#     squeue -u $USER
#     tail -f logs/output/simvp_single_sample_<jobid>.out
#
# 任务：用 GOES-16 ABI ch13 数据做 SimVP 单样本训练测试（2 帧输入 -> 1 帧输出），
#       目的是先把 SimVP 整条链路跑通，而不是评测泛化能力。

set -euo pipefail

# ---- 可按需修改 ----
CONDA_ENV=cira-diff-cb
WORKDIR=/home/group1/26fall_aiclass/cb/cira-diff/reproduction/simvp/single_sample_test
OPENSTL_DIR="$WORKDIR/OpenSTL"
CONDA_BASE=/home/group1/miniconda3
DATA_ROOT=/data1/satcast
ZARR_NAME=edm_GOES_ch13_train_dataset.zarr
SAMPLE_IDX=0          # 单样本测试用训练集里的第几条样本
EPOCHS=2000
EX_NAME=satcast_simvp_single_sample
# --------------------

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
export OMP_NUM_THREADS=4

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
echo "[INFO] data         : $DATA_ROOT/$ZARR_NAME (sample #$SAMPLE_IDX)"
echo "[INFO] gpu (CUDA_VISIBLE_DEVICES) : ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "[INFO] gpu name     : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[INFO] start time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

# SimVP 的 2 帧输入 / 1 帧输出由 configs/satcast/SimVP.py +
# openstl/datasets/dataset_constant.py['satcast'] 共同决定；
# 单样本由 config 里的 single_sample_idx 控制（train/val/test 共用同一条样本）。
set +e
python -u tools/train.py \
    -d satcast \
    -m SimVP \
    -c configs/satcast/SimVP.py \
    --data_root "$DATA_ROOT" \
    -b 1 \
    -vb 1 \
    --num_workers 2 \
    -e "$EPOCHS" \
    --res_dir "$WORKDIR/outputs" \
    --ex_name "$EX_NAME"
STATUS=$?
set -e

# 训练结束后做推理评估：加载 best.ckpt，和 persistence 基线对比并出图。
# （OpenSTL 的 trainer.test() 只给 mse/mae 两个数字，没有对照，单样本测试
#   需要一个基线才能说明模型确实学到了东西。）
if [ "$STATUS" -eq 0 ]; then
    echo "======================================================================"
    echo "[INFO] 开始单样本评估（加载 best.ckpt + persistence 基线对比）"
    echo "======================================================================"
    set +e
    python -u "$WORKDIR/eval_single_sample.py" \
        --ckpt "$WORKDIR/outputs/$EX_NAME/checkpoints/best.ckpt" \
        --config "$OPENSTL_DIR/configs/satcast/SimVP.py" \
        --data_root "$DATA_ROOT" \
        --zarr_name "$ZARR_NAME" \
        --idx "$SAMPLE_IDX" \
        --out_dir "$WORKDIR/outputs/$EX_NAME"
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
