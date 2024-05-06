#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=performance
#SBATCH --out=slurm/logs/messis-lightning.ipynb_out.txt
#SBATCH --err=slurm/logs/messis-lightning.ipynb_err.txt
#SBATCH --job-name="Run messis-lightning.ipynb"


# For testing add before each command:
# srun --partition performance

poetry shell

dvc pull
dvc repro

papermill scripts/messis-lightning.ipynb scripts/messis-lightning.output.ipynb
