#!/bin/bash
#SBATCH --job-name=vanilla_unet_full
#SBATCH --output=/home/group1/26fall_aiclass/ly/cira-diff/outputs/slurm_logs/vanilla_unet_full_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/ly/cira-diff/outputs/slurm_logs/vanilla_unet_full_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

mkdir -p /home/group1/26fall_aiclass/ly/cira-diff/outputs/slurm_logs
mkdir -p /home/group1/26fall_aiclass/ly/cira-diff/outputs/vanilla_unet_full/logs/images

cd /home/group1/26fall_aiclass/ly/cira-diff

source /home/group1/miniconda3/etc/profile.d/conda.sh
conda activate cira-diff-ly

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1

python -u scripts/Chase_2025/train_vanilla_unet_Chase2025.py

echo "FINISHED at $(date)"