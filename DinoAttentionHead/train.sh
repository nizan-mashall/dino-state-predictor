#!/bin/bash
#SBATCH --job-name=dino-train
#SBATCH --partition=dlc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err


source ~/dino-state-predictor/myenv/bin/activate
cd ~/dino-state-predictor/DinoAttentionHead

python dinoAttention_train.py \
    --hdf5_path ~/dino-state-predictor/demo_with_images.hdf5 \
    --checkpoint_dir ~/dino-state-predictor/DinoAttentionHead/ckp

