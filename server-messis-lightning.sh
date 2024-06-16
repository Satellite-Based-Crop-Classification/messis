#!/bin/sh

# Create the logs directory if it doesn't exist
mkdir -p logs/server

# Get the current date and time for the logfile
timestamp=$(date +'%Y-%m-%d_%H-%M-%S')
logfile="logs/server/output_${timestamp}.log"

# Run each command and redirect their output to the logfile
{
  echo "Running dvc pull..."
  poetry run dvc pull

  echo "Running dvc repro..."
  poetry run dvc repro

  echo "Converting Jupyter notebook to script..."
  poetry run jupyter nbconvert --to script model_training.ipynb

  echo "Running model training script..."
  nohup poetry run python model_training.py > "$logfile" 2>&1 &  # Essential to use srun for multi-GPU training!

  # For debugging, use papermill to run the notebook and see print statements
  # poetry run papermill model_training.ipynb model_training.output.ipynb
} > "$logfile" 2>&1 &

# Print the logfile name to the console
echo "Training is running in the background. Output is logged to: $logfile"