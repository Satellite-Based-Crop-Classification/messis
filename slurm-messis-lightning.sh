#!/bin/sh
#SBATCH --time=08:00:00
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --partition=performance
#SBATCH --out=logs/slurm/model_training.ipynb_out.txt
#SBATCH --err=logs/slurm/model_training.ipynb_out.txt
#SBATCH --job-name="messis"
# To exclude: SBATCH --exclude=gpu22a

# Config above according to: https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html

# For testing add before each command:
# srun --partition performance

# Create the directory for SLURM log files if it does not exist
mkdir -p logs/slurm

poetry run dvc pull
poetry run dvc repro

poetry run jupyter nbconvert --to script model_training.ipynb
poetry run srun python model_training.py  # Essential to use srun for multi-GPU training!

# For debugging, use papermill to run the notebook and see print statements
# poetry run srun papermill model_training.ipynb model_training.output.ipynb