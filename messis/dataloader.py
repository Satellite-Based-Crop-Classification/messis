import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule

import os
import re
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
    def __init__(self, data_dir, fold_indicies, transform_img=None, transform_mask=None, transform_field_ids=None, debug=False, subset_size=None):
        self.data_dir = data_dir
        self.chips_dir = os.path.join(data_dir, 'chips')
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.transform_field_ids = transform_field_ids
        self.debug = debug
        self.images = []
        self.masks = []
        self.field_ids = []

        self.means, self.stds = self.load_stats(fold_indicies, 3) # Hardcoded 3 timesteps for now
        self.transform_img_load = self.get_img_load_transforms(self.means, self.stds)
        self.transform_mask_load = self.get_mask_load_transforms()
        self.transform_field_ids_load = self.get_field_ids_load_transforms()
        
        # Adjust file selection based on fold
        for file in os.listdir(self.chips_dir):
            if re.match(f".*_fold_[{"".join(fold_indicies)}]_merged.tif", file):
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

    def load_stats(self, fold_indicies, n_timesteps=3):
        """Load normalization statistics for dataset from YAML file."""
        stats_path = os.path.join(self.data_dir, 'chips_stats.yaml')
        if self.debug:
            print(f"Loading mean/std stats from {stats_path}")
        assert os.path.exists(stats_path), f"mean/std stats file for dataset not found at {stats_path}"
        with open(stats_path, 'r') as file:
            stats = yaml.safe_load(file)
        mean_list, std_list, n_list = [], [], []
        for fold in fold_indicies:
            key = f'fold_{fold}'
            if key not in stats:
                raise ValueError(f"mean/std stats for fold {fold} not found in {stats_path}")
            if self.debug:
                print(f"Stats with selected test fold {fold}: {stats[key]} over {n_timesteps} timesteps.")
            mean_list.append(stats[key]['mean']) # list of 6 means
            std_list.append(stats[key]['std']) # list of 6 stds
            n_list.append(stats[key]['n_chips']) # list of 6 ns
        # aggregate means and stds over all folds
        means, stds = [], []
        for channel in range(mean_list[0].shape[0]):
            means.append(torch.stack([mean_list[i][channel] for i in range(len(mean_list))]).mean())
            # stds are waaaay more complex to aggregate
            # \sqrt{\frac{\sum_{i=1}^{n} (\sigma_i * (n_i - 1))}{\sum_{i=1}^{n} (n_i) - n}}
            variances = torch.stack([std_list[i][channel] ** 2 for i in range(len(std_list))])
            n = torch.tensor([n_list[i] for i in range(len(n_list))], dtype=torch.float32)
            combined_variance = torch.sum(variances * (n - 1)) / (torch.sum(n) - len(n_list))
            stds.append(torch.sqrt(combined_variance))

        # make means and stds into 2d arrays, as torchvision would otherwise convert it into a 3d tensor which is incompatible with our 4d temporal images
        # https://github.com/pytorch/vision/blob/6e18cea3485066b7277785415bf2e0422dbdb9da/torchvision/transforms/_functional_tensor.py#L923
        return means * n_timesteps, stds * n_timesteps

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
    def __init__(self, data_dir, train_folds, val_folds, test_folds, batch_size=8, num_workers=4, debug=False, subsets=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug
        self.subsets = subsets if subsets is not None else {}
        
        GeospatialDataModule.validate_folds(train_folds, val_folds, test_folds)
        self.train_folds = train_folds
        self.val_folds = val_folds
        self.test_folds = test_folds

        # NOTE: Transforms on this level not used for now
        self.transform_img = get_img_transforms()
        self.transform_mask = get_mask_transforms()

    @staticmethod
    def validate_folds(train, val, test):
        if train is None or val is None or test is None:
            raise ValueError("All fold sets must be specified")
        
        if len(set(train) & set(val)) > 0 or len(set(train) & set(test)) > 0 or len(set(val) & set(test)) > 0:
            raise ValueError("Folds must be mutually exclusive")

    def setup(self, stage=None):
        common_params = {
            'data_dir': self.data_dir,
            'debug': self.debug,
        }
        if stage in ('fit', None):
            self.train_dataset = GeospatialDataset(fold_indicies=self.train_folds, subset_size=self.subsets.get('train', None), **common_params)
            self.val_dataset   = GeospatialDataset(fold_indicies=self.val_folds,   subset_size=self.subsets.get('val',   None), **common_params)
        if stage in ('test', None):
            self.test_dataset  = GeospatialDataset(fold_indicies=self.test_folds,  subset_size=self.subsets.get('test',  None), **common_params)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)