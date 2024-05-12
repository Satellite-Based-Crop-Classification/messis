# Messis

Crop classification model for the Swiss cantons of Zurich and Thurgau, built upon the OpenMMLab framework and fine-tuning the Prithvi-100M model.

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

Set up this environment to run Prithvi with the MMCV/MMSegmentation framework.
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