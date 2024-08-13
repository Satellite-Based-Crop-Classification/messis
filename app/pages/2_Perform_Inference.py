import streamlit as st
import leafmap.foliumap as leafmap
from transformers import PretrainedConfig
from folium import Icon

from messis.messis import Messis
from app.inference import perform_inference

st.set_page_config(layout="wide")

GEOTIFF_PATH = "../data/stacked_features.tif"

# Load the model
@st.cache_resource
def load_model():
    config = PretrainedConfig.from_pretrained('crop-classification/messis', revision='47d9ca4')
    model = Messis.from_pretrained('crop-classification/messis', cache_dir='./hf_cache/', revision='47d9ca4')
    return model, config
model, config = load_model()

def perform_inference_step():
    st.title("Step 2: Perform Inference")

    st.sidebar.header("Settings")

    if "selected_location" not in st.session_state:
        st.error("No location selected. Please select a location first.")
        st.page_link("pages/1_Select_Location.py", label="Select Location", icon="📍")
        return

    lat, lon = st.session_state["selected_location"]
    st.write(f"Using POI: Latitude {lat}, Longitude {lon}")

    # Timestep Slider
    timestep = st.sidebar.slider("Select Timestep", 1, 9, 5)

    # Band Dropdown
    band_options = {
        "RGB": [1, 2, 3],  # Adjust indices based on the actual bands in your GeoTIFF
        "NIR": [4],
        "SWIR1": [5],
        "SWIR2": [6]
    }
    vmin_vmax = { 
        "RGB": (89, 1878),
        "NIR": (165, 5468),
        "SWIR1": (120, 3361),
        "SWIR2": (94, 2700)
    }
    selected_band = st.sidebar.selectbox("Select Satellite Band to Display", options=list(band_options.keys()), index=0)
    
    # Calculate the band indices based on the selected timestep
    selected_bands = [band + (timestep - 1) * 6 for band in band_options[selected_band]]
    
    # Initialize the map
    m = leafmap.Map(center=(lat, lon), zoom=10, draw_control=False)

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
    poi_icon = Icon(color="red", prefix="fa", icon="crosshairs")
    m.add_marker(location=[lat, lon], popup="POI", layer_name="POI", icon=poi_icon)

    # Display the map in the Streamlit app
    m.to_streamlit()

if __name__ == "__main__":
    perform_inference_step()
