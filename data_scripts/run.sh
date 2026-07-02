#!/bin/bash
#SBATCH --job-name=data-add-images
#SBATCH --partition=dlc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/down_%j.out
#SBATCH --error=logs/down_%j.err


source ~/dino-state-predictor/myenv/bin/activate

python -u data_scripts/add_images_transport.py
