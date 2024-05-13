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
    dvc remote modify gdrive --local gdrive_service_account_json_file_path service-account-yvo.json
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
