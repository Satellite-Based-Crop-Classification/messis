import random
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
    def __init__(self, data_dir, test_fold, train=True, transform_img=None, transform_mask=None, transform_field_ids=None, crop_to=None, debug=False, subset_size=None):
        self.data_dir = data_dir
        self.chips_dir = os.path.join(data_dir, 'chips')
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.transform_field_ids = transform_field_ids
        self.crop_to = crop_to
        self.debug = debug
        self.train = train
        self.images = []
        self.masks = []
        self.field_ids = []

        self.means, self.stds = self.load_stats(test_fold)
        self.transform_img_load = self.get_img_load_transforms(self.means, self.stds)
        self.transform_mask_load = self.get_mask_load_transforms()
        self.transform_field_ids_load = self.get_field_ids_load_transforms()
        
        # Adjust file selection based on fold
        for file in os.listdir(self.chips_dir):
            if file.endswith("_merged.tif") and f"_fold_{test_fold}" not in file if train else file.endswith("_merged.tif") and f"_fold_{test_fold}" in file:
                self.images.append(file)
                mask_file = file.replace("_merged.tif", "_mask.tif")
                self.masks.append(mask_file)
                field_ids_file = file.replace("_merged.tif", "_field_ids.tif")
                self.field_ids.append(field_ids_file)

        assert len(self.images) == len(self.masks), "Number of images and masks do not match"

        # If subset_size is specified, randomly select a subset of the data
        if subset_size is not None and len(self.images) > subset_size:
            print(f"Randomly selecting {subset_size} samples from {len(self.images)} samples.")
            selected_indices = random.sample(range(len(self.images)), subset_size)
            self.images = [self.images[i] for i in selected_indices]
            self.masks = [self.masks[i] for i in selected_indices]
            self.field_ids = [self.field_ids[i] for i in selected_indices]

    def load_stats(self, test_fold):
        """Load normalization statistics for dataset from YAML file."""
        stats_path = os.path.join(self.data_dir, 'chips_stats.yaml')
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
        
    def get_field_ids_load_transforms(self):
        return transforms.Compose([
            ToTensorTransform(torch.uint16)
        ])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.chips_dir, self.images[idx])
        mask_path = os.path.join(self.chips_dir, self.masks[idx])
        field_ids_path = os.path.join(self.chips_dir, self.field_ids[idx])
        
        img = rasterio.open(img_path).read().astype('uint16')
        mask = rasterio.open(mask_path).read().astype('uint8')
        field_ids = rasterio.open(field_ids_path).read().astype('uint16')
        
        # Apply our base transforms
        img = self.transform_img_load(img)
        mask = self.transform_mask_load(mask)
        field_ids = self.transform_field_ids_load(field_ids)
        

        # Crop the image and mask if applicable
        if self.crop_to:
            if self.train:
                i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.crop_to, self.crop_to))
                img = transforms.functional.crop(img, i, j, h, w)
                mask = transforms.functional.crop(mask, i, j, h, w)
                field_ids = transforms.functional.crop(field_ids, i, j, h, w)
            else:
                img = transforms.functional.center_crop(img, self.crop_to)
                mask = transforms.functional.center_crop(mask, self.crop_to)
                field_ids = transforms.functional.center_crop(field_ids, self.crop_to)

        # Apply additional transforms passed from module if applicable
        if self.transform_img is not None:
            img = self.transform_img(img)
        if self.transform_mask is not None:
            mask = self.transform_mask(mask)
        if self.transform_field_ids is not None:
            field_ids = self.transform_field_ids(field_ids)

        targets_tier1 = mask[0, :, :].type(torch.long)
        targets_tier2 = mask[1, :, :].type(torch.long)
        targets_tier3 = mask[2, :, :].type(torch.long)
        
        return img, ((targets_tier1, targets_tier2, targets_tier3), field_ids)

class GeospatialDataModule(LightningDataModule):
    def __init__(self, data_dir, test_fold, batch_size=8, num_workers=4, crop_to=None, debug=False, subsets=None):
        super().__init__()
        self.data_dir = data_dir
        self.test_fold = test_fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_to = crop_to
        self.debug = debug
        self.subsets = subsets if subsets is not None else {}

        # NOTE: Transforms on this level not used for now
        self.transform_img = get_img_transforms()
        self.transform_mask = get_mask_transforms()

    def setup(self, stage=None):
        common_params = {
            'data_dir': self.data_dir,
            'test_fold': self.test_fold,
            'crop_to': self.crop_to,
            'debug': self.debug,
        }
        if stage in ('fit', None):
            self.train_dataset = GeospatialDataset(train=True,  subset_size=self.subsets.get('train', None), **common_params)
            self.val_dataset   = GeospatialDataset(train=False, subset_size=self.subsets.get('val',   None), **common_params)
        if stage in ('test', None):
            self.test_dataset  = GeospatialDataset(train=False, subset_size=self.subsets.get('test',  None), **common_params)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)