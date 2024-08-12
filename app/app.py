import geopandas as gpd
import streamlit as st
import leafmap.foliumap as leafmap

st.set_page_config(layout="wide")

def app():
    st.title("GeoJSON Viewer")

    # Load the area of interest (AOI) from the uploaded GeoJSON file
    aoi = gpd.read_file("./data/aoi.geojson")

    # Load GeoJSON files
    zh_gdf = gpd.read_file("./data/fields_zh_filtered.geojson")

    # Load satellite image
    geotiff_path = "../data/stacked_features.tif"

    # Select columns 
    zh_gdf = zh_gdf[["GEMEINDE", "FLAECHE", "NUTZUNG", "NUTZUNGSCO", "JAHR", "geometry"]]

    # Create checkboxes to activate/deactivate the GeoJSON layers
    show_zh = st.sidebar.checkbox("Show ZH Fields", True)

    # Add a slider for timestep selection
    timestep = st.sidebar.slider("Select Timestep", 1, 9, 5)  # Default timestep is 1

    # Add a dropdown for band selection
    band_options = {
        "RGB": [1, 2, 3],  # Adjust indices based on the actual bands in your GeoTIFF
        "NIR": [4],
        "SWIR1": [5],
        "SWIR2": [6]
    }
    selected_band = st.sidebar.selectbox("Select Band to Display", options=list(band_options.keys()), index=0)  # Default is RGB

    # vmin vmax per band option
    vmin_vmax = { # Values are based on the actual pixel values in the stacked_features GeoTIFF
        "RGB": (89, 1878),
        "NIR": (165, 5468),
        "SWIR1": (120, 3361),
        "SWIR2": (94, 2700)
    }

    # Calculate the band indices based on the selected timestep
    selected_bands = [band + (timestep - 1) * 6 for band in band_options[selected_band]]
    print(selected_bands)

    # Initialize the map
    m = leafmap.Map(center=(47.5, 8.5), zoom=10)

    # Add the GeoDataFrames to the map based on user selection
    if show_zh:
        m.add_gdf(
            zh_gdf, 
            layer_name="ZH Fields",
            random_color_column="NUTZUNGSCO",
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

    # Display the map in the Streamlit app
    m.to_streamlit(width=950, height=600)

# Run the app
if __name__ == "__main__":
    app()