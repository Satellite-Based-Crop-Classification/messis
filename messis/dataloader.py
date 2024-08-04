import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule
import os
import re
import yaml
import rasterio
import dvc.api


params = dvc.api.params_show()
N_TIMESTEPS = params['number_of_timesteps']

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
        if data.shape[0] != N_TIMESTEPS * 6:
            raise ValueError(f"Expected {N_TIMESTEPS*6} channels, got {data.shape[1]}")
        
        # Step 1: Reshape the data to group the N_TIMESTEPS*6 bands into N_TIMESTEPS groups of 6 bands
        data = data.view(N_TIMESTEPS, 6, height, width)
        
        # Step 2: Permute to bring the bands to the front
        data = data.permute(1, 0, 2, 3)  # NOTE: Prithvi wants it bands first # after this, shape is (6, N_TIMESTEPS, height, width)
        return data

class RandomFlipAndJitterTransform:
    """
    Apply random horizontal and vertical flips, and channel jitter to the input image and corresponding mask.

    Parameters:
    -----------
    flip_prob : float, optional (default=0.5)
        Probability of applying horizontal and vertical flips to the image and mask.
        Each flip (horizontal and vertical) is applied independently based on this probability.

    jitter_std : float, optional (default=0.02)
        Standard deviation of the Gaussian noise added to the image channels for jitter.
        This value controls the intensity of the random noise applied to the image channels.

    Effects of Parameters:
    ----------------------
    flip_prob:
        - Higher flip_prob increases the likelihood of the image and mask being flipped.
        - A value of 0 means no flipping, while a value of 1 means always flip.

    jitter_std:
        - Higher jitter_std increases the intensity of the noise added to the image channels.
        - A value of 0 means no noise, while larger values add more significant noise.
    """
    def __init__(self, flip_prob=0.5, jitter_std=0.02):
        self.flip_prob = flip_prob
        self.jitter_std = jitter_std

    def __call__(self, img, mask, field_ids):
        # Shapes (..., H, W)| img: torch.Size([6, N_TIMESTEPS, 224, 224]), mask: torch.Size([N_TIMESTEPS, 224, 224]), field_ids: torch.Size([1, 224, 224])
        
        # Temporarily convert field_ids to int32 for flipping (flip not implemented for uint16)
        field_ids = field_ids.to(torch.int32)

        # Random horizontal flip
        if random.random() < self.flip_prob:
            img = torch.flip(img, [2])
            mask = torch.flip(mask, [1])
            field_ids = torch.flip(field_ids, [1])

        # Random vertical flip
        if random.random() < self.flip_prob:
            img = torch.flip(img, [3])
            mask = torch.flip(mask, [2])
            field_ids = torch.flip(field_ids, [2])

        # Convert field_ids back to uint16
        field_ids = field_ids.to(torch.uint16)

        # Channel jitter
        noise = torch.randn(img.size()) * self.jitter_std
        img += noise

        return img, mask, field_ids

def get_img_transforms():
    return transforms.Compose([])

def get_mask_transforms():
    return transforms.Compose([])

class GeospatialDataset(Dataset):
    def __init__(self, data_dir, fold_indicies, transform_img=None, transform_mask=None, transform_field_ids=None, debug=False, subset_size=None, data_augmentation=None):
        self.data_dir = data_dir
        self.chips_dir = os.path.join(data_dir, 'chips')
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.transform_field_ids = transform_field_ids
        self.debug = debug
        self.images = []
        self.masks = []
        self.field_ids = []
        self.data_augmentation = data_augmentation

        self.means, self.stds = self.load_stats(fold_indicies, N_TIMESTEPS)
        self.transform_img_load = self.get_img_load_transforms(self.means, self.stds)
        self.transform_mask_load = self.get_mask_load_transforms()
        self.transform_field_ids_load = self.get_field_ids_load_transforms()
        
        # Adjust file selection based on fold
        for file in os.listdir(self.chips_dir):
            if re.match(f".*_fold_[{''.join([str(f) for f in fold_indicies])}]_merged.tif", file):
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
            mean_list.append(torch.Tensor(stats[key]['mean'])) # list of 6 means
            std_list.append(torch.Tensor(stats[key]['std'])) # list of 6 stds
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

        # Apply additional transforms passed from GeospatialDataModule if applicable
        if self.transform_img is not None:
            img = self.transform_img(img)
        if self.transform_mask is not None:
            mask = self.transform_mask(mask)
        if self.transform_field_ids is not None:
            field_ids = self.transform_field_ids(field_ids)

        # Apply data augmentation if enabled
        if self.data_augmentation is not None and self.data_augmentation.get('enabled', True):
            img, mask, field_ids = RandomFlipAndJitterTransform(
                flip_prob=self.data_augmentation.get('flip_prob', 0.5), 
                jitter_std=self.data_augmentation.get('jitter_std', 0.02)
            )(img, mask, field_ids)

        # Load targets for given tiers
        num_tiers = mask.shape[0]
        targets = ()
        for i in range(num_tiers):
            targets += (mask[i, :, :].type(torch.long),)
        
        return img, (targets, field_ids)

class GeospatialDataModule(LightningDataModule):
    def __init__(self, data_dir, train_folds, val_folds, test_folds, batch_size=8, num_workers=4, debug=False, subsets=None, data_augmentation=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug
        self.subsets = subsets if subsets is not None else {}
        self.data_augmentation = data_augmentation if data_augmentation is not None else {}
        
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
        print(f"Setting up GeospatialDataModule for stage: {stage}. Data augmentation config: {self.data_augmentation}")
        common_params = {
            'data_dir': self.data_dir,
            'debug': self.debug,
            'data_augmentation': self.data_augmentation
        }
        common_params_val_test = {
            **common_params,
             'data_augmentation': {
                'enabled': False
            }
        }
        if stage in ('fit', None):
            self.train_dataset = GeospatialDataset(fold_indicies=self.train_folds, subset_size=self.subsets.get('train', None), **common_params)
            self.val_dataset   = GeospatialDataset(fold_indicies=self.val_folds,   subset_size=self.subsets.get('val',   None), **common_params_val_test)
        if stage in ('test', None):
            self.test_dataset  = GeospatialDataset(fold_indicies=self.test_folds,  subset_size=self.subsets.get('test',  None), **common_params_val_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
