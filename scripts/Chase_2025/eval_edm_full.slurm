#!/bin/bash
#SBATCH --job-name=eval_edm_full
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=outputs/slurm_logs/eval_edm_full_%j.out
#SBATCH --error=outputs/slurm_logs/eval_edm_full_%j.err
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /home/group1/miniconda3/etc/profile.d/conda.sh
conda activate cira-diff-ly
cd /home/group1/26fall_aiclass/ly/cira-diff

mkdir -p outputs/slurm_logs outputs/eval_edm_full

echo "=========================================="
echo "EDM Full Evaluation: EXP-011 (287ep) vs Official (1000ep)"
echo "Partition: debug, GPU: 1x4090, Mem: 32G"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csvnoheader | head -1)"
echo "GPU Free Mem: $(nvidia-smi --query-gpu=memory.free --format=csvnoheader | head -1)"
echo "Started at: $(date)"
echo "=========================================="

python -u scripts/Chase_2025/eval_edm_comparison.py \
  --ckpt-a /home/group1/26fall_aiclass/ly/cira-diff/outputs/edm_full/ \
  --ckpt-a-format diffusers \
  --ckpt-b /data1/satcast/edm_plain_diffusion/edm_plain_diffusion/checkpoint.pth \
  --ckpt-b-format pth \
  --name-a "EXP-011 EDM (287ep)" \
  --name-b "Official EDM (1000ep)" \
  --test-zarr /data1/satcast/edm_GOES_ch13_test_dataset.zarr \
  --outdir outputs/eval_edm_full \
  --ens-size 10 \
  --max-single-step 1024 \
  --max-rollout 128 \
  --base-seed 0 2>&1

EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
echo "Finished at: $(date)"
exit $EXIT_CODE