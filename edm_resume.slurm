#!/bin/bash
#SBATCH --job-name=edm_resume
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/group1/26fall_aiclass/ly/cira-diff/outputs/slurm_logs/edm_resume_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/ly/cira-diff/outputs/slurm_logs/edm_resume_%j.err

export WANDB_MODE=disabled
export TENSORBOARD_CACHE_ENABLE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/group1/26fall_aiclass/ly/cira-diff

echo "=========================================="
echo "  EDM Resume Training (auto-resume)"
echo "  Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Start: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
source /home/group1/miniconda3/etc/profile.d/conda.sh
conda activate cira-diff-ly
python -u scripts/Chase_2025/train_edm_Chase2025.py
echo ""
echo "=========================================="
echo "  EDM Resume Done: $(date)"
echo "=========================================="
