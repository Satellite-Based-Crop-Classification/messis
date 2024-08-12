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

    # Load satellite image
    geotiff_path = "../data/stacked_features.tif"

    # Select columns 
    zh_gdf = zh_gdf[["GEMEINDE", "FLAECHE", "NUTZUNG", "NUTZUNGSCO", "JAHR", "geometry"]]

    # Create checkboxes to activate/deactivate the GeosJSON layers
    show_zh = st.sidebar.checkbox("Show ZH Fields", True)

    # Initialize the map
    m = leafmap.Map(center=(47.5, 8.5), zoom=10)

    # Add the GeoDataFrames to the map based on user selection
    if show_zh:
        m.add_gdf(
            zh_gdf, 
            layer_name="ZH Fields",
            random_color_column="NUTZUNGSCO",
        )

        # Add the GeoTIFF to the map
        m.add_raster(
            geotiff_path,
            layer_name="Satellite Image",
            bands=[31, 32, 33],  # Specify the bands: [Red, Green, Blue]
            fit_bounds=True,  # Fit the map bounds to the raster,
            vmin=50,  # Adjust this value to increase brightness
            vmax=2000,  # Adjust this value to control the upper range
        )

    # Display the map in the Streamlit app
    m.to_streamlit(width=950, height=600)

# Run the app
if __name__ == "__main__":
    app()
