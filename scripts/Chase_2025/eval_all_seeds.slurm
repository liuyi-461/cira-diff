#!/bin/bash
#SBATCH --job-name=eval_all_seeds
#SBATCH --output=/home/group1/26fall_aiclass/ly/cira-diff/outputs/slurm_logs/eval_all_seeds_%j.out
#SBATCH --error=/home/group1/26fall_aiclass/ly/cira-diff/outputs/slurm_logs/eval_all_seeds_%j.err
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

cd /home/group1/26fall_aiclass/ly/cira-diff
source /home/group1/miniconda3/etc/profile.d/conda.sh
conda activate cira-diff-ly

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python -u /tmp/eval_all_seeds.py

echo "FINISHED at $(date)"
