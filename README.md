# Messis - Crop Classification Model

`Messis` is a crop classification model for the agricultural landscapes of Switzerland. It is built upon the geospatial foundation model [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M), which was originally pre-trained on U.S. satellite data. Messis has been trained using our ZueriCrop 2.0 dataset, a collection of Sentinel-2 imagery combined with ground-truth crop labels that covers agricultural regions in Switzerland.

<img src="./assets/messis.jpeg" alt="Messis" width="600">

The Messis model leverages a three-tier hierarchical label structure, optimized for remote sensing tasks, to enhance its classification accuracy across different crop types. By adapting Prithvi to the specific challenges of Swiss agriculture—such as smaller field sizes and higher image resolutions by the Sentinel-2 satellites—Messis demonstrates the versatility of pretrained geospatial models in handling new downstream tasks.

Additionally, Messis reduces the need for extensive labeled data by effectively utilizing Prithvi's pretrained weights. In evaluations, Messis achieved a notable F1 score of 34.8% across 48 crop classes.

## Key Features

1. **Adapted for High-Resolution Crop Classification:** Messis is fine-tuned from the Prithvi geospatial foundation model, originally trained on U.S. data, and optimized for high-resolution Sentinel-2 imagery specific to Swiss agricultural landscapes.
2. **Leveraged Hierarchical Label Structure:** Utilizes a remote-sensing-focused hierarchical label structure, enabling more accurate classification across multiple levels of crop granularity.
3. **Pretrained Weight Utilization:** Demonstrated significant performance improvement by leveraging Prithvi's pretrained weights, achieving a doubled F1 score compared to training from scratch.
4. **Dataset:** Trained on the ZueriCrop 2.0 dataset, which features higher image dimension (224x224 pixels) compared to the original ZueriCrop dataset.

## Documentation

- The poster for our model can be found [here](./assets/Poster.pdf).
- Read our full report here [here](./assets/BAT-Report_Satellite-based-Crop-Classification.pdf).

## Repository Structure

The repository is structured as follows, with the most important files and directories highlighted:

```markdown
└── 📁messis
    └── README.md
    └── pyproject.toml [ℹ️ Poetry configuration file]
    └── params.yaml [ℹ️ DVC configuration file]
    └── model_training.ipynb [ℹ️ Jupyter notebook for training the model]
    └── server-messis-lightning.sh [ℹ️ Script for training the model on a server with GPU]
    └── slurm-messis-lightning.sh [ℹ️ SLURM script for training the model on a cluster]
    └── .env.example [ℹ️ Example environment file]
    └── 📁assets [ℹ️ Assets created for our report]
    └── 📁data [ℹ️ The directory DVC uses to store data]
    └── 📁messis [ℹ️ Full implementation of the Messis model]
    └── 📁prithvi [ℹ️ Code for the Prithvi model, adapted from https://github.com/NASA-IMPACT/hls-foundation-os/]
    └── 📁notebooks [ℹ️ Various notebooks for exploration, experimentation and evaluation]
```

## Usage

Experience the Messis model firsthand by trying it out in our interactive [Hugging Face Spaces Demo](https://huggingface.co/spaces/crop-classification/messis-demo).

To learn how to load the model and perform inference, check the [source code](https://huggingface.co/spaces/crop-classification/messis-demo/tree/main) in our Huggingface Space.

## Setup

Install poetry and the dependencies:

```bash
poetry install
```

Note if you're using Windows: You need to reinstall torch and torchvision with CUDA support. Change `cu121` to your CUDA version and check whether the versions of torch and torchvision match with the ones in the `pyproject.toml` file. For more details see: [https://github.com/python-poetry/poetry/issues/6409](https://github.com/python-poetry/poetry/issues/6409)

```bash
poetry shell
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121 -U
pip install torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121 -U
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

1. Initialize DVC:
```bash
dvc remote modify --local ssh password request-the-password-from-the-team
```

1. Pull the data:
```bash
dvc pull
```

## MMCV Environment Setup (optional)

Only set up this environment if you want to run Prithvi with the MMCV/MMSegmentation framework (see `prithvi` folder).

This environment is as described in `hls-foundation-os`:

```bash
conda create --name hls-foundation-os python==3.9
conda activate hls-foundation-os
pip install torch==1.11.0 torchvision==0.12.0
pip install -e .
pip install -U openmim
mim install mmcv-full==1.6.2
```

Next, download the Pritvhi model using the `download_prithvi_100M.ipynb` notebook in the `prithvi/model` directory.

##  SLURM Setup

1. Clone Repo

    ```bash
    srun --partition performance git clone  https://github.com/Satellite-Based-Crop-Classification/messis.git
    ```

2. Install Poetry

    ```bash
    srun --partition performance curl -sSL https://install.python-poetry.org | python3 -
    ```

3. Open the `.bashrc` file in a text editor, such as nano or vim. For example:

   ```bash
   nano ~/.bashrc
   ```

4. Add the following line at the end of the file:

   ```bash
   export PATH="/home2/YOUR_USER/.local/bin:$PATH"
   ```

5. To make the changes effective immediately in your current session, source the `.bashrc` file:

   ```bash
   source ~/.bashrc
   ```

6. Install the dependencies

    ```bash
    srun --partition performance poetry install
    ```

7. Enter the virtual environment

    ```bash
    poetry shell
    ```

8. Configure DVC

    ```bash
    dvc remote modify --local ssh password request-the-password-from-the-team
    ```

9. Pull the data

    ```bash
    srun --partition performance dvc pull
    ```

10. Log in to W&B

    ```bash
    wandb login <API_KEY>
    ```

11. Log into Hugging Face

    ```bash
    huggingface-cli login
    ```

12. Configure git user

    ```bash
    git config --global user.name "Name Surname"
    git config --global user.email "your.mail@example.com"
    ```

## Training

Resources for optimizing the training:

- https://archive.is/ELPqJ
- https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html
- PyTorch Lightning SLURM: https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html

### Train on Server with GPU

- Start Training Job: `sh server-messis-lightning.sh`

When you want to stop the job, you can kill the entire process group:

1. **Find the process group ID (PGID)**:

   ```sh
   ps -o pgid,cmd -p $(pgrep -f 'server-messis-lightning.sh')
   ```

2. **Kill the process group**:

   ```sh
   kill -TERM -<PGID>
   ```

   Replace `<PGID>` with the actual process group ID you found.

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

## Debugging on Remote Server

1. Start debug server on your remote server: 

    For GPU Server:
    ```bash
    python -m debugpy --listen 0.0.0.0:5678 --wait-for-client model_training.py`
    ```

    For SLURM (untested):
    ```bash
    srun --partition performance poetry run python -m debugpy --listen
    ```

2. Launch the "Remote Attach" debug configuration in your VS Code (see `.vscode/launch.json`). VS Code will connect to the debug server on the remote server and you can debug as usual.
