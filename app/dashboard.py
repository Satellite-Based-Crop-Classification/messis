import folium
import geopandas as gpd
import streamlit as st
import leafmap.foliumap as leafmap
import rasterio
# from transformers import PretrainedConfig
from messis.messis import Messis  # Assuming you have a custom model class
import numpy as np
from folium import Map, Marker, Icon
from folium.elements import MacroElement
from jinja2 import Template
from streamlit_js_eval import streamlit_js_eval
import torch

st.set_page_config(layout="wide")

GEOTIFF_PATH = "../data/stacked_features.tif"

# Load the model
# config['hparams']['backbone_weights_path'] = 'huggingface'
# config = PretrainedConfig.from_pretrained('crop-classification/messis', revision='47d9ca4')
model = Messis.from_pretrained('crop-classification/messis', cache_dir='./hf_cache/', revision='47d9ca4')

def run_inference_on_window(window_data):
    """
    Run the inference on the extracted window data.
    """
    inputs = np.expand_dims(window_data, axis=0)  # Add batch dimension
    inputs = torch.tensor(inputs).float()
    print(inputs.shape)
    outputs = model(inputs)
    return outputs

def extract_window(geotiff_path, lon, lat):
    """
    Extract a 224x224 window centered on the clicked coordinates (lon, lat).
    """
    with rasterio.open(geotiff_path) as src:
        # Convert the lat/lon to row/col
        row, col = src.index(lon, lat)
        
        # Define the window around the clicked point
        half_window_size = 112
        window = rasterio.windows.Window(
            col_off=col - half_window_size,
            row_off=row - half_window_size,
            width=224,
            height=224
        )
        
        # Read all bands over the 9 timesteps (6 bands * 9 timesteps = 54 layers)
        window_data = src.read(window=window)
        
        return window_data
    
def run_inference(lat, lon):
    """
    Run the inference on the clicked coordinates (lon, lat).
    """
    # Extract the window around the clicked coordinates
    window_data = extract_window(GEOTIFF_PATH, lon, lat)
    
    # Run the inference on the extracted window
    outputs = run_inference_on_window(window_data)
    
    # Get the predicted class
    predicted_class = np.argmax(outputs)
    
    # Display the predicted class
    st.write(f"Predicted Class: {predicted_class}")

@st.cache_data
def uploaded_file_to_gdf(data):
    import tempfile
    import os
    import uuid

    _, file_extension = os.path.splitext(data.name)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{file_extension}")

    gdf = gpd.read_file(file_path)
    return gdf

def app():
    st.title("Messis - Crop Classification")

    # Load the area of interest (AOI) from the uploaded GeoJSON file
    aoi = gpd.read_file("./data/aoi.geojson")

    # Load GeoJSON files
    zh_gdf = gpd.read_file("./data/fields_zh_filtered.geojson")
    predictions = gpd.read_file("./data/predictions.geojson")

    # Load satellite image
    geotiff_path = "../data/stacked_features.tif"

    # Select columns 
    zh_gdf = zh_gdf[["GEMEINDE", "FLAECHE", "NUTZUNG", "NUTZUNGSCO", "JAHR", "geometry"]]

    # Create checkboxes to activate/deactivate the GeoJSON layers
    show_zh = st.sidebar.checkbox("Show ZH Fields", True)

    # Add a slider for timestep selection
    timestep = st.sidebar.slider("Select Timestep", 1, 9, 5)  # Default timestep is 5

    # Add a dropdown for band selection
    band_options = {
        "RGB": [1, 2, 3],  # Adjust indices based on the actual bands in your GeoTIFF
        "NIR": [4],
        "SWIR1": [5],
        "SWIR2": [6]
    }
    selected_band = st.sidebar.selectbox("Select Band to Display", options=list(band_options.keys()), index=0)  # Default is RGB

    # streamlit textfield for coordinates
    poi_coords = st.sidebar.text_input("POI Coordinates", "8.635254,47.381789")
    lon, lat = [float(coord) for coord in poi_coords.split(",")]
    poi_icon = Icon(color="red", prefix="fa", icon="crosshairs")
    # run_inference(lat, lon)

    # vmin vmax per band option
    vmin_vmax = { # Values are based on the actual pixel values in the stacked_features GeoTIFF
        "RGB": (89, 1878),
        "NIR": (165, 5468),
        "SWIR1": (120, 3361),
        "SWIR2": (94, 2700)
    }

    # Calculate the band indices based on the selected timestep
    selected_bands = [band + (timestep - 1) * 6 for band in band_options[selected_band]]
    print(selected_bands)

    data = st.file_uploader(
        "Upload a GeoJSON file to use as an ROI. 👇",
        type=["geojson"],
    )

    # Initialize the map
    m = leafmap.Map(center=(47.5, 8.5), zoom=10, draw_control=True, draw_export=False)

    # Add the GeoDataFrames to the map based on user selection
    m.add_gdf(
        zh_gdf, 
        layer_name="ZH Fields"
    )

    # m.add_gdf(
    #     predictions, 
    #     layer_name="Predictions",
    #     style_callback=lambda feat: {"fillColor": "#ff7800", "backgroundColor": "#ff7800", "color": "#ff7800", "opacity": 0.8}
    # )

    m.add_data(predictions,
       layer_name = "Predictions",
       column="correct",
       add_legend=False,
       style_function=lambda x: {"fillColor": "green" if x["properties"]["correct"] == True else "red", "color": "black", "weight": 0, "fillOpacity": 0.25},
    )

    # Add the GeoTIFF to the map with the selected bands and timestep
    m.add_raster(
        geotiff_path,
        layer_name="Satellite Image",
        bands=selected_bands,  # Specify the bands based on user selection
        fit_bounds=True,  # Fit the map bounds to the raster,
        vmin=vmin_vmax[selected_band][0],  # Adjust this value to control the lower range
        vmax=vmin_vmax[selected_band][1],  # Adjust this value to control the upper range
    )

    # Show the POI on the map
    m.add_marker(location=[lat, lon], popup="Point of Interest", layer_name="POI", icon=poi_icon)

    if data:
        gdf = uploaded_file_to_gdf(data)
        try:
            print(gdf)
            st.session_state["poi"] = gdf.geometry.centroid.x[0], gdf.geometry.centroid.y[0]
            m.add_gdf(gdf, "POI")
        except Exception as e:
            st.error(e)
            st.error("Please draw another ROI and try again.")
            return

    # Display the map in the Streamlit app
    m.to_streamlit()

if __name__ == "__main__":
    app()