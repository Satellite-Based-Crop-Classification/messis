# Messis

Hierarchical crop classification model for the Swiss cantons of Zurich and Thurgau, built upon the Prithvi-100M Geospatial model as backbone.

## Setup

Install poetry and the dependencies:

```bash
poetry install
```

Make sure you set VSCode Setting `python.venvPath` to your poetry venv path, so that you can select the virtual environment in VSCode.

To enter the virtual environment:

```bash
poetry shell
```

To install new packages, use the following command, but make sure you have exited the shell with `exit` before:

```bash
poetry add <package>
```

Setup DVC:

1. Get a service account JSON file from Google Cloud Platform and place it in the `.dvc` directory.

2. Initialize DVC:
```bash
dvc remote modify gdrive --local gdrive_service_account_json_file_path .dvc/service-account.json
```

3. Pull the data:
```bash
dvc pull
```
### MMCV Environment Setup (optional)

Set up this environment to run Prithvi with the MMCV/MMSegmentation framework (see `prithvi` folder).

This environment is as described in `hls-foundation-os`:

```bash
conda create --name hls-foundation-os python==3.9
conda activate hls-foundation-os
pip install torch==1.11.0 torchvision==0.12.0
pip install -e .
pip install -U openmim
mim install mmcv-full==1.6.2
```

Next, download the Pritvhi model using the [download](./prithvi/model/download.ipynb) notebook in the `prithvi/model` directory.

##  SLURM Setup

1. Clone Repo

    ```bash
    srun --partition top6 git clone --recurse-submodules -j8 https://<PAT>@github.com/Satellite-Based-Crop-Classification/messis.git
    ```

2. Install Poetry

    ```bash
    srun --partition top6 curl -sSL https://install.python-poetry.org | python3 -
    ```

3. Open the `.bashrc` file in a text editor, such as nano or vim. For example:

   ```bash
   nano ~/.bashrc
   ```

4. Add the following line at the end of the file:

   ```bash
   export PATH="/home2/yvo/.local/bin:$PATH"
   ```

5. To make the changes effective immediately in your current session, source the `.bashrc` file:

   ```bash
   source ~/.bashrc
   ```

6. Install the dependencies

    ```bash
    srun --partition top6 poetry install
    ```

7. Enter the virtual environment

    ```bash
    poetry shell
    ```

8. Configure DVC

    ```bash
    scp ./service-account-yvo.json yvo@slurmlogin.cs.technik.fhnw.ch:/home2/yvo/code/messis/.dvc/
    ```

    ```bash
    dvc remote modify gdrive gdrive_use_service_account true
    dvc remote modify gdrive --local gdrive_service_account_json_file_path .dvc/service-account-yvo.json
    ```

9. Pull the data

    ```bash
    srun --partition top6 dvc pull
    ```

10. Log in to W&B

    ```bash
    wandb login <API_KEY>
    ```

11. Configure git user

    ```bash
    git config --global user.name "Yvo Keller"
    git config --global user.email "hi@yvo.ai"
    ```

## Training

Resources for optimizing the training:
- https://archive.is/ELPqJ
- https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html
- PyTorch Lightning SLURM: https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html

### Train on Server with GPU

- Start Training Job: `sh server-messis-lightning.sh`

### Train on SLURM

Slurm Commands:
- Start SLURM Training Job: `sbatch slurm-messis-lightning.sh`
- Show SLURM Jobs: `squeue`
- Show SLURM Cluster nodes with GPU Info: `scontrol show nodes`
- Cancel SLURM Job: `scancel job_id`

### Use Multi-GPU Training on SLURM

Make sure the config in `messis-lightning.sh` and `model_training.ipynb` are correctly set up and have the same values.

See the parameters --nodes, --gres and -ntasks-per-node (--gres and --ntasks-per-node must match) in the SLURM script:

```bash
#!/bin/sh
#SBATCH --time=08:00:00
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=...)
#SBATCH --partition=performance
#SBATCH --out=slurm/logs/model_training.ipynb_out.txt
#SBATCH --err=slurm/logs/model_training.ipynb_out.txt
#SBATCH --job-name="messis"
```

The counterpart in the notebook, see num_nodes and devices, must match the SLURM script:

```python
trainer = Trainer(
    logger=wandb_logger,
    log_every_n_steps=1,
    callbacks=[
        LogMessisMetrics  (hparams, params['paths']['dataset_info'], debug=False),
        LogConfusionMatrix(hparams, params['paths']['dataset_info'], debug=False),
        early_stopping
    ],
    accumulate_grad_batches=hparams['accumulate_grad_batches'],  # Gradient accumulation
    max_epochs=hparams['max_epochs'],
    accelerator="gpu",
    strategy="ddp",         # Use Distributed Data Parallel
    num_nodes=1,            # Number of nodes
    devices=2,              # Number of GPUs to use
    precision='16-mixed'    # Train with 16-bit precision (https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision)
)
```

Then, make sure you are starting the python script with `srun` in `messis-lightning.sh`:

```bash
poetry run srun python model_training.py  # Essential to use srun for multi-GPU training!
```

To start the training, run `sbatch messis-lightning.sh` in the terminal of the SLURM login node.