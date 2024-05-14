#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --partition=performance
#SBATCH --ntasks-per-node=1
#SBATCH --out=slurm/logs/model_training.ipynb_out.txt
#SBATCH --err=slurm/logs/model_training.ipynb_err.txt
#SBATCH --job-name="messis.train"


# For testing add before each command:
# srun --partition performance

# Create the directory for SLURM log files if it does not exist
mkdir -p slurm/logs

poetry run dvc pull
poetry run dvc repro

poetry run papermill model_training.ipynb model_training.output.ipynb