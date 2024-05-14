#!/bin/sh
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --partition=performance
#SBATCH --out=slurm/logs/model_training.ipynb_out.txt
#SBATCH --err=slurm/logs/model_training.ipynb_out.txt
#SBATCH --job-name="messis"

# Config above according to: https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html

# For testing add before each command:
# srun --partition performance

# Create the directory for SLURM log files if it does not exist
mkdir -p slurm/logs

poetry run dvc pull
poetry run dvc repro

poetry run jupyter nbconvert --to script model_training.ipynb
poetry run python model_training.py

# poetry run papermill model_training.ipynb model_training.output.ipynb