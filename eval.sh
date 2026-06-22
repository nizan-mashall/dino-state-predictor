#!/bin/bash
#SBATCH --job-name=dino-eval
#SBATCH --partition=dlc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

source ~/dino-state-predictor/myenv/bin/activate
cd ~/dino-state-predictor
python -u tinybc_test.py