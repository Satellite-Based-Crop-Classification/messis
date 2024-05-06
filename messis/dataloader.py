import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule

import os
import yaml
import rasterio

class ToTensorTransform(object):
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, data):
        return torch.tensor(data, dtype=self.dtype)
    
class NormalizeTransform(object):
    def __init__(self, means, stds):
        self.mean = means
        self.std = stds

    def __call__(self, data):
        return transforms.Normalize(self.mean, self.std)(data)
    
class PermuteTransform:
    def __call__(self, data):
        height, width = data.shape[-2:]

        # Ensure the channel dimension is as expected
        if data.shape[0] != 18:
            raise ValueError(f"Expected 18 channels, got {data.shape[1]}")
        
        # Step 1: Reshape the data to group the 18 bands into 3 groups of 6 bands
        data = data.view(3, 6, height, width)
        
        # Step 2: Permute to bring the bands to the front
        data = data.permute(1, 0, 2, 3)  # NOTE: Prithvi wants it bands first
        return data
    
def get_img_transforms():
    # TODO: Think about possible data augmentation techniques that could work for our data
    return transforms.Compose([])

def get_mask_transforms():
    return transforms.Compose([])

class GeospatialDataset(Dataset):
    def __init__(self, data_dir, test_fold, train=True, transform_img=None, transform_mask=None, crop_to=None, debug=False):
        self.data_dir = data_dir
        self.chips_dir = os.path.join(data_dir, 'chips')
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.crop_to = crop_to
        self.debug = debug
        self.train = train
        self.images = []
        self.masks = []

        self.means, self.stds = self.load_stats(test_fold)
        self.transform_img_load = self.get_img_load_transforms(self.means, self.stds)
        self.transform_mask_load = self.get_mask_load_transforms()
        
        # Adjust file selection based on fold
        for file in os.listdir(self.chips_dir):
            if file.endswith("_merged.tif") and f"_fold_{test_fold}" not in file if train else file.endswith("_merged.tif") and f"_fold_{test_fold}" in file:
                self.images.append(file)
                mask_file = file.replace("_merged.tif", "_mask.tif")
                self.masks.append(mask_file)

    def load_stats(self, test_fold):
        """Load normalization statistics for dataset from YAML file."""
        stats_path = os.path.join(self.data_dir, 'chips_fold_stats.yaml')
        if self.debug:
            print(f"Loading mean/std stats from {stats_path}")
        assert os.path.exists(stats_path), f"mean/std stats file for dataset not found at {stats_path}"
        with open(stats_path, 'r') as file:
            stats = yaml.safe_load(file)
        key = f'fold_{test_fold}'
        if key not in stats:
            raise ValueError(f"mean/std stats for fold {test_fold} not found in {stats_path}")
        if self.debug:
            print(f"Stats with selected test fold {test_fold}: {stats[key]} over {stats[key]['n_timesteps']} timesteps.")
        return stats[key]['mean'] * stats[key]['n_timesteps'], stats[key]['std'] * stats[key]['n_timesteps']

    def get_img_load_transforms(self, means, stds):
        return transforms.Compose([
            ToTensorTransform(torch.float32),
            NormalizeTransform(means, stds),
            PermuteTransform()
        ])

    def get_mask_load_transforms(self):
        return transforms.Compose([
            ToTensorTransform(torch.uint8)
        ])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.chips_dir, self.images[idx])
        mask_path = os.path.join(self.chips_dir, self.masks[idx])
        
        img = rasterio.open(img_path).read().astype('uint16')
        mask = rasterio.open(mask_path).read().astype('uint8')
        
        # Apply our base transforms
        img = self.transform_img_load(img)
        mask = self.transform_mask_load(mask)

        # Crop the image and mask if applicable
        if self.crop_to:
            if self.train:
                i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.crop_to, self.crop_to))
                img = transforms.functional.crop(img, i, j, h, w)
                mask = transforms.functional.crop(mask, i, j, h, w)
            else:
                img = transforms.functional.center_crop(img, self.crop_to)
                mask = transforms.functional.center_crop(mask, self.crop_to)

        # Apply additional transforms passed from module if applicable
        if self.transform_img is not None:
            img = self.transform_img(img)
        if self.transform_mask is not None:
            mask = self.transform_mask(mask)

        targets_tier1 = mask[0, :, :]
        targets_tier2 = mask[1, :, :]
        targets_tier3 = mask[2, :, :]
        
        return img, (targets_tier1, targets_tier2, targets_tier3)

class GeospatialDataModule(LightningDataModule):
    def __init__(self, data_dir, test_fold, batch_size=8, num_workers=4, crop_to=None, debug=False):
        super().__init__()
        self.data_dir = data_dir
        self.test_fold = test_fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_to = crop_to

        # NOTE: Transforms on this level not used for now
        self.transform_img = get_img_transforms()
        self.transform_mask = get_mask_transforms()

        self.debug = debug

    def setup(self, stage=None):
        if stage in ('fit', None):
            self.train_dataset = GeospatialDataset(self.data_dir, self.test_fold, train=True, crop_to=self.crop_to, debug=self.debug)
            self.val_dataset = GeospatialDataset(self.data_dir, self.test_fold, train=False, crop_to=self.crop_to, debug=self.debug)
        if stage in ('test', None):
            self.test_dataset = GeospatialDataset(self.data_dir, self.test_fold, train=False, crop_to=self.crop_to, debug=self.debug)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)