#!/bin/bash
#SBATCH --job-name=eval_gan
#SBATCH --output=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/eval_gan_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/test_dl/logs/eval_gan_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# GAN 训练结束后自动：
#   1) 用统一协议评估（test 全量 1024 条 × 18 步）
#   2) 导出为 pred_official.npy 同格式 (1024,18,256,256)
#   3) 对比 GAN 前后逐时效 RMSE 与**锐度比**
#
# 提交：sbatch --dependency=afterany:<训练jobid> test_dl/eval_gan.slurm

set -euo pipefail

CONDA_ENV=openstl
CONDA_BASE=/home/group1/miniconda3
OPENSTL_DIR=/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL
LOGDIR="$OPENSTL_DIR/test_dl/logs"
DATA_ROOT=/home/group1/26fall_aiclass/yr/data1/data1/satcast
CONFIG="$OPENSTL_DIR/configs/satcast/SimVP.py"
GAN_DIR="$OPENSTL_DIR/work_dirs/satcast_simvp_gan"
CKPT="$GAN_DIR/checkpoints/best.ckpt"

mkdir -p "$LOGDIR"
cd "$OPENSTL_DIR"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$OPENSTL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"
mkdir -p "$MPLCONFIGDIR"

echo "[INFO] job id : ${SLURM_JOB_ID:-N/A}"

if [ ! -f "$CKPT" ]; then
    echo "[FATAL] 未找到 GAN checkpoint：$CKPT（训练可能未完成）"
    exit 3
fi

echo "================= 1) 统一协议评估（1024 条 × 18 步） ================="
python -u "$OPENSTL_DIR/tools/eval_simvp_baseline.py" \
    --ckpt "$CKPT" --config "$CONFIG" --data_root "$DATA_ROOT" \
    --out_dir "$GAN_DIR/baseline_eval_rollout18_full" \
    --rollout-steps 18 --max-samples 1024 --batch_size 16 \
    --norm-mean 0 --norm-std 1

echo "================= 2) 导出为 pred_official 同格式 ================="
python -u "$OPENSTL_DIR/tools/export_preds_official_format.py" \
    --extra gan=satcast_simvp_gan

echo "================= 3) GAN 前后锐度对比 ================="
python -u - <<'PY'
import numpy as np, torch
def gm(x):
    dy=torch.abs(x[:,:,1:,:]-x[:,:,:-1,:]); dx=torch.abs(x[:,:,:,1:]-x[:,:,:,:-1])
    return ((dy.mean()+dx.mean())/2).item()
base='work_dirs/satcast_simvp_full/baseline_eval_rollout18_full/'
gan='work_dirs/satcast_simvp_gan/baseline_eval_rollout18_full/'
b=np.load(base+'preds.npy',mmap_mode='r')
try:
    g=np.load(gan+'preds.npy',mmap_mode='r')
except Exception as e:
    print('无法读取 GAN 预测：',e); raise SystemExit(0)
t=np.load(base+'trues.npy',mmap_mode='r')
print('  LT   baseline锐度比   GAN锐度比   变化')
for j in [0,2,5,11,17]:
    tr=torch.from_numpy(np.array(t[j],dtype=np.float32))
    rb=gm(torch.from_numpy(np.array(b[j],dtype=np.float32)))/gm(tr)
    rg=gm(torch.from_numpy(np.array(g[j],dtype=np.float32)))/gm(tr)
    print('  %2d      %.3f         %.3f     %+.3f'%(j+1,rb,rg,rg-rb))
PY

echo "[DONE] GAN 评估完成"
