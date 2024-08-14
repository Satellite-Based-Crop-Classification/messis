import os
import torch
import yaml
import json
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from pyproj import Transformer
from torchvision import transforms
import numpy as np
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd

from messis.messis import LogConfusionMatrix

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
            if self.debug:
                print("Source Transform", src.transform)
                print(f"UTM X: {utm_x}, UTM Y: {utm_y}")
            
            try:
                px, py = rowcol(src.transform, utm_x, utm_y)
            except ValueError:
                raise ValueError("Coordinates out of bounds for this raster.")
            
            if self.debug:
                print(f"Row: {py}, Column: {px}")
            
            half_window_size = self.window_size // 2
            
            row_off = px - half_window_size
            col_off = py - half_window_size
            
            if row_off < 0:
                row_off = 0
            if col_off < 0:
                col_off = 0
            if row_off + self.window_size > src.width:
                row_off = src.width - self.window_size
            if col_off + self.window_size > src.height:
                col_off = src.height - self.window_size
            
            window = Window(col_off, row_off, self.window_size, self.window_size)
            window_transform = src.window_transform(window)
            if self.debug:
                print(f"Window: {window}")
                print(f"Window Transform: {window_transform}")
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
    
def crop_predictions_to_gdf(field_ids, targets, predictions, transform, crs, class_names):
    """
    Convert field_ids, targets, and predictions tensors to field polygons with corresponding class reference.
    
    :param field_ids: PyTorch tensor of shape (1, 224, 224) representing individual fields
    :param targets: PyTorch tensor of shape (1, 224, 224) for targets
    :param predictions: PyTorch tensor of shape (1, 224, 224) for predictions
    :param transform: Affine transform for georeferencing
    :param crs: Coordinate reference system (CRS) of the data
    :param class_names: Dictionary mapping class indices to class names
    :return: GeoPandas DataFrame with polygons, prediction class labels, and target class labels
    """
    field_array = field_ids.squeeze().cpu().numpy().astype(np.int32)
    target_array = targets.squeeze().cpu().numpy().astype(np.int8)
    pred_array = predictions.squeeze().cpu().numpy().astype(np.int8)

    polygons = []
    field_values = []
    target_values = []
    pred_values = []

    for geom, field_value in shapes(field_array, transform=transform):
        polygons.append(shape(geom))
        field_values.append(field_value)

        # Get a single value from the field area for targets and predictions
        target_value = target_array[field_array == field_value][0]
        pred_value = pred_array[field_array == field_value][0]
        
        target_values.append(target_value)
        pred_values.append(pred_value)

    gdf = gpd.GeoDataFrame({
        'geometry': polygons,
        'field_id': field_values,
        'target': target_values,
        'prediction': pred_values
    }, crs=crs)

    gdf['prediction_class'] = gdf['prediction'].apply(lambda x: class_names[x])
    gdf['target_class'] = gdf['target'].apply(lambda x: class_names[x])

    gdf['correct'] = gdf['target'] == gdf['prediction']

    gdf = gdf[gdf.geometry.area > 250] # Threshold for small polygons

    return gdf

def perform_inference(lon, lat, model, config, debug=False):
    features_path = "../data/stacked_features.tif"
    labels_path = "../data/labels.tif"
    field_ids_path = "../data/field_ids.tif"
    stats_path = "../data/chips_stats.yaml"

    loader = InferenceDataLoader(features_path, labels_path, field_ids_path, stats_path, n_timesteps=9, fold_indices=[0], debug=True)

    # Coordinates must be in EPSG:4326 and lon lat order - are converted to the CRS of the raster
    satellite_data, label_data, field_ids_data, features_transform, features_crs = loader.get_data(lon, lat)

    if debug:
        # Print the shape of the extracted data
        print(satellite_data.shape)
        print(label_data.shape)
        print(field_ids_data.shape)

    with open('../data/dataset_info.json', 'r') as file:
        dataset_info = json.load(file)
    class_names = dataset_info['tier3']

    tiers_dict = {k: v for k, v in config.hparams.get('heads_spec').items() if v.get('is_metrics_tier', False)}
    tiers = list(tiers_dict.keys())

    # Perform inference
    model.eval()
    with torch.no_grad():
        output = model(satellite_data)['tier3_refinement_head']

    pixelwise_outputs_stacked, majority_outputs_stacked = LogConfusionMatrix.get_pixelwise_and_majority_outputs(output, tiers, field_ids=field_ids_data, dataset_info=dataset_info)
    majority_tier3_predictions = majority_outputs_stacked[2] # Tier 3 predictions

    # Convert the predictions to a GeoDataFrame
    gdf = crop_predictions_to_gdf(field_ids_data, label_data, majority_tier3_predictions, features_transform, features_crs, class_names)

    # Simple GeoDataFrame with only the necessary columns
    gdf = gdf[['prediction_class', 'target_class', 'correct', 'geometry']]
    gdf.columns = ['Prediction', 'Target', 'Correct', 'geometry']
    # gdf = gdf[gdf['Target'] != 'Background']

    return gdf