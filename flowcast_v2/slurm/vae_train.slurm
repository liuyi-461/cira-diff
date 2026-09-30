#!/bin/bash
#SBATCH --job-name=vae_train
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
WORKDIR=${WORKDIR:-${SLURM_SUBMIT_DIR:?Submit from the flowcast directory or set WORKDIR}}
cd "$WORKDIR"
# Activate your training environment before sbatch. Slurm inherits it.
: "${TRAIN_FILE:?Set an independent training Zarr}"
: "${VAL_FILE:?Set an independent validation Zarr}"
: "${CONFIG:?Set the run-specific YAML with fitted normalization}"
: "${OUTPUT_DIR:?Set a unique run output directory}"
IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
NGPU=${NUM_GPUS:-${#GPU_LIST[@]}}
python -m torch.distributed.run --standalone --nproc_per_node="$NGPU" \
    --module experiments.goes.autoencoder.dist_train_autoencoder_kl \
    --config "$CONFIG" --train_file "$TRAIN_FILE" --val_file "$VAL_FILE" \
    --output_dir "$OUTPUT_DIR"
