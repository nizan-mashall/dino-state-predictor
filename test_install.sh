#!/bin/bash
#SBATCH --job-name=test-install
#SBATCH --partition=dlc
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/test_%j.out

source ~/dino-state-predictor/myenv/bin/activate

python -c "import mujoco; print('MuJoCo:', mujoco.__version__)"
python -c "import robosuite; print('Robosuite:', robosuite.__version__)"
