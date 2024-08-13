import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from pyproj import Transformer
from torchvision import transforms
from transformers import PretrainedConfig

import numpy as np
import torch
import yaml
import os

from messis.messis import Messis

class InferenceDataLoader:
    def __init__(self, features_path, labels_path, field_ids_path, stats_path, window_size=224, n_timesteps=3, fold_indices=None, debug=False):
        self.features_path = features_path
        self.labels_path = labels_path
        self.field_ids_path = field_ids_path
        self.stats_path = stats_path
        self.window_size = window_size
        self.n_timesteps = n_timesteps
        self.fold_indices = fold_indices if fold_indices is not None else []
        self.debug = debug
        
        # Load normalization stats
        self.means, self.stds = self.load_stats()

        # Set up the transformer for coordinate conversion
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)

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

    def identify_window(self, path, lon, lat):
        """Identify the 224x224 window centered on the clicked coordinates (lon, lat) from the specified GeoTIFF."""
        with rasterio.open(path) as src:
            # Transform the coordinates from WGS84 to UTM (EPSG:32632)
            utm_x, utm_y = self.transformer.transform(lon, lat)
            
            try:
                px, py = rowcol(src.transform, utm_x, utm_y)
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
            window_transform = src.window_transform(window)
            crs = src.crs

            return window, window_transform, crs

    def extract_window(self, path, window):
        """Extract data from the specified window from the GeoTIFF."""
        with rasterio.open(path) as src:
            window_data = src.read(window=window)

        if self.debug:
            print(f"Extracted window data from {path}")
            print(f"Min: {window_data.min()}, Max: {window_data.max()}")
        
        return window_data

    def prepare_data_for_model(self, features_data):
        """Prepare the window data for model inference."""
        # Convert to tensor
        features_data = torch.tensor(features_data, dtype=torch.float32)
        
        # Normalize
        normalize = transforms.Normalize(mean=self.means, std=self.stds)
        features_data = normalize(features_data)
        
        # Permute the dimensions if needed
        height, width = features_data.shape[-2:]
        features_data = features_data.view(self.n_timesteps, 6, height, width).permute(1, 0, 2, 3)
        
        # Add batch dimension
        features_data = features_data.unsqueeze(0)
        
        return features_data

    def get_data(self, lon, lat):
        """Extract, normalize, and prepare data for inference, including labels and field IDs."""
        # Identify the window and get the georeferencing information
        window, features_transform, features_crs = self.identify_window(self.features_path, lon, lat)
        
        # Extract data from the GeoTIFF, labels, and field IDs
        features_data = self.extract_window(self.features_path, window)
        label_data = self.extract_window(self.labels_path, window)
        field_ids_data = self.extract_window(self.field_ids_path, window)
        
        # Prepare the window data for the model
        prepared_features_data = self.prepare_data_for_model(features_data)
        
        # Convert labels and field_ids to tensors (without normalization)
        label_data = torch.tensor(label_data, dtype=torch.long)
        field_ids_data = torch.tensor(field_ids_data, dtype=torch.long)
        
        # Return the prepared data along with transform and CRS
        return prepared_features_data, label_data, field_ids_data, features_transform, features_crs
    
