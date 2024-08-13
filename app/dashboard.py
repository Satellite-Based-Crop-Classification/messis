import streamlit as st
import leafmap.foliumap as leafmap
from transformers import PretrainedConfig
from folium import Icon

from messis.messis import Messis
from app.inference import perform_inference

st.set_page_config(layout="wide")

GEOTIFF_PATH = "../data/stacked_features.tif"

# Load the model
# config['hparams']['backbone_weights_path'] = 'huggingface'
@st.cache_resource
def load_model():
    config = PretrainedConfig.from_pretrained('crop-classification/messis', revision='47d9ca4')
    model = Messis.from_pretrained('crop-classification/messis', cache_dir='./hf_cache/', revision='47d9ca4')
    return model, config
model, config = load_model()

def app():
    st.title("Messis 🌾 - Crop Classification 🌎")

    # Instructions
    st.sidebar.header("Instructions")
    st.sidebar.write("1. Select the timestep and band to view the satellite image.")
    st.sidebar.write("2. Enter POI coordinates or draw a point on the map.")
    st.sidebar.write("3. Click 'Run Inference' to get predictions.")

    # Settings
    st.sidebar.header("Settings")

    # Timestep Slider
    timestep = st.sidebar.slider("Select Timestep", 1, 9, 5)  # Default timestep is 5

    # Band Dropdown
    band_options = {
        "RGB": [1, 2, 3],  # Adjust indices based on the actual bands in your GeoTIFF
        "NIR": [4],
        "SWIR1": [5],
        "SWIR2": [6]
    }
    vmin_vmax = { # Values are based on the actual pixel values in the stacked_features GeoTIFF
        "RGB": (89, 1878),
        "NIR": (165, 5468),
        "SWIR1": (120, 3361),
        "SWIR2": (94, 2700)
    }
    selected_band = st.sidebar.selectbox("Select Satellite Band to Display", options=list(band_options.keys()), index=0)  # Default is RGB

    # Calculate the band indices based on the selected timestep
    selected_bands = [band + (timestep - 1) * 6 for band in band_options[selected_band]]
    print('Selected Bands', selected_bands)

    # Coordinates for the Point of Interest (POI)
    poi_coords = st.sidebar.text_input("POI Coordinates", "8.635254,47.381789")
    lon, lat = [float(coord) for coord in poi_coords.split(",")]
    poi_icon = Icon(color="red", prefix="fa", icon="crosshairs")

    # Initialize the map
    m = leafmap.Map(center=(47.5, 8.5), zoom=10, draw_control=True, draw_export=False)

    # Perform inference
    if st.button("Run Inference"):
        predictions = perform_inference(lon, lat, model, config, debug=True)
        m.add_data(predictions,
            layer_name = "Predictions",
            column="correct",
            add_legend=False,
            style_function=lambda x: {"fillColor": "green" if x["properties"]["correct"] else "red", "color": "black", "weight": 0, "fillOpacity": 0.25},
        )
        st.success("Inference completed!")

    # GeoTIFF Satellite Imagery with selected timestep and band
    m.add_raster(
        GEOTIFF_PATH,
        layer_name="Satellite Image",
        bands=selected_bands,
        fit_bounds=True,
        vmin=vmin_vmax[selected_band][0],
        vmax=vmin_vmax[selected_band][1],
    )

    # Show the POI on the map
    m.add_marker(location=[lat, lon], popup="POI", layer_name="POI", icon=poi_icon)

    drawn_features = m.draw_features
    if drawn_features:
        print(drawn_features)

    # Display the map in the Streamlit app
    m.to_streamlit()

if __name__ == "__main__":
    app()