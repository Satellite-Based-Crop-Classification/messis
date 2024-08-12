import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from torchvision import transforms

from transformers import PretrainedConfig
from messis.messis import Messis

import torch
import yaml
import os

class InferenceDataLoader:
    def __init__(self, geotiff_path, stats_path, window_size=224, n_timesteps=3, fold_indices=None, debug=False):
        self.geotiff_path = geotiff_path
        self.stats_path = stats_path
        self.window_size = window_size
        self.n_timesteps = n_timesteps
        self.fold_indices = fold_indices if fold_indices is not None else []
        self.debug = debug
        
        # Load normalization stats
        self.means, self.stds = self.load_stats()

    def load_stats(self):
        """Load normalization statistics for dataset from YAML file."""
        if self.debug:
            print(f"Loading mean/std stats from {self.stats_path}")
        assert os.path.exists(self.stats_path), f"Mean/std stats file not found at {self.stats_path}"

        with open(self.stats_path, 'r') as file:
            stats = yaml.safe_load(file)

        mean_list, std_list, n_list = [], [], []
        for fold in self.fold_indices:
            key = f'fold_{fold}'
            if key not in stats:
                raise ValueError(f"Mean/std stats for fold {fold} not found in {self.stats_path}")
            if self.debug:
                print(f"Stats with selected test fold {fold}: {stats[key]} over {self.n_timesteps} timesteps.")
            mean_list.append(torch.tensor(stats[key]['mean'])) # list of 6 means
            std_list.append(torch.tensor(stats[key]['std'])) # list of 6 stds
            n_list.append(stats[key]['n_chips']) # list of 6 ns
        
        means, stds = [], []
        for channel in range(mean_list[0].shape[0]):
            means.append(torch.stack([mean_list[i][channel] for i in range(len(mean_list))]).mean())
            variances = torch.stack([std_list[i][channel] ** 2 for i in range(len(std_list))])
            n = torch.tensor([n_list[i] for i in range(len(n_list))], dtype=torch.float32)
            combined_variance = torch.sum(variances * (n - 1)) / (torch.sum(n) - len(n_list))
            stds.append(torch.sqrt(combined_variance))
        
        return means * self.n_timesteps, stds * self.n_timesteps

    def extract_window(self, lon, lat):
        """Extract a 224x224 window centered on the clicked coordinates (lon, lat)."""
        with rasterio.open(self.geotiff_path) as src:
            try:
                px, py = rowcol(src.transform, lon, lat)
            except ValueError:
                raise ValueError("Coordinates out of bounds for this raster.")
            
            if self.debug:
                print(f"Row: {py}, Column: {px}")
            
            half_window_size = self.window_size // 2
            
            col_off = px - half_window_size
            row_off = py - half_window_size
            
            if col_off < 0:
                col_off = 0
            if row_off < 0:
                row_off = 0
            if col_off + self.window_size > src.width:
                col_off = src.width - self.window_size
            if row_off + self.window_size > src.height:
                row_off = src.height - self.window_size
            
            window = Window(col_off, row_off, self.window_size, self.window_size)
            window_data = src.read(window=window)
            
            return window_data

    def prepare_data_for_model(self, window_data):
        """Prepare the window data for model inference."""
        # Convert to tensor
        window_data = torch.tensor(window_data, dtype=torch.float32)
        
        # Normalize
        normalize = transforms.Normalize(mean=self.means, std=self.stds)
        window_data = normalize(window_data)
        
        # Permute the dimensions if needed
        height, width = window_data.shape[-2:]
        window_data = window_data.view(self.n_timesteps, 6, height, width).permute(1, 0, 2, 3)
        
        # Add batch dimension
        window_data = window_data.unsqueeze(0)
        
        return window_data

    def get_data(self, lon, lat):
        """Extract, normalize, and prepare data for inference."""
        window_data = self.extract_window(lon, lat)
        prepared_data = self.prepare_data_for_model(window_data)
        return prepared_data

# Example usage:
geotiff_path = "../data/stacked_features.tif"
stats_path = "../data/chips_stats.yaml"
loader = InferenceDataLoader(geotiff_path, stats_path, n_timesteps=9, fold_indices=[0])

# Get data for a specific latitude and longitude
prepared_data = loader.get_data(8.759193, 47.373942)

print(prepared_data.shape)