import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import pandas as pd
from pyproj import Proj, Transformer
import matplotlib.pyplot as plt

# Define the coordinates (degrees)
latitude, longitude = 47.365638125, 8.82493713 # Hittnau

# Convert coordinates from degrees to UTM
proj_wgs84 = Proj('epsg:4326')
proj_utm = Proj('epsg:32632')
transformer = Transformer.from_proj(proj_wgs84, proj_utm)

east, north = transformer.transform(latitude, longitude)

# Define a 10x10 km bounding box
half_size = 5000  # 10 km / 2 in meters
minx, miny = east - half_size, north - half_size
maxx, maxy = east + half_size, north + half_size
bounding_box = box(minx, miny, maxx, maxy)

# Load field polygons
fields_zh_gdf = gpd.read_file('../data/ZH_2019_LW_NUTZUNGSFLAECHEN_F.shp')
fields_tg_gdf = gpd.read_file('../data/TG_2019_Nutzungsflaechen_2019.gpkg')

# Filter for the bounding box region
fields_zh_gdf = fields_zh_gdf[fields_zh_gdf.intersects(bounding_box)]
fields_tg_gdf = fields_tg_gdf[fields_tg_gdf.intersects(bounding_box)]

# Combine the dataframes
fields_gdf = pd.concat([fields_zh_gdf, fields_tg_gdf], ignore_index=True)

# Load stacked features metadata
stacked_features = rasterio.open('../data/stacked_features.tif')
transform = stacked_features.transform

# Function to get 224x224 image chip for a given field
def get_image_chip(field_geom, size=224):
    # Get the bounding box of the field
    minx, miny, maxx, maxy = field_geom.bounds
    # Calculate the center of the bounding box
    centerx, centery = (minx + maxx) / 2, (miny + maxy) / 2
    # Calculate the pixel coordinates of the center
    pixelx, pixely = ~transform * (centerx, centery)
    # Calculate the pixel coordinates of the top-left corner of the chip
    startx, starty = int(pixelx - size / 2), int(pixely - size / 2)
    # Extract the chip for each band in the selected timestep
    chip = stacked_features.read(window=((starty, starty + size), (startx, startx + size)))
    return chip

# Function to run inference on the image chip (dummy function for this example)
def run_inference(image_chip):
    # Dummy inference: just return a random crop type for now
    crop_types = ['Wheat', 'Corn', 'Soy', 'Barley']
    return np.random.choice(crop_types)

# Create a base map using the default folium tiles
map_center = [latitude, longitude]
m = folium.Map(
    location=map_center,
    zoom_start=14,
)

# Add field polygons to the map
for _, field in fields_gdf.iterrows():
    folium.GeoJson(field['geometry'], name='geojson').add_to(m)

# Display the map in Streamlit
st.title('Crop Classification Map')
st.write('Select a field to run inference on the crop type.')
map_data = st_folium(m, width=700, height=500)

# Handle field selection
if map_data and 'last_active_drawing' in map_data:
    selected_field = map_data['last_active_drawing']
    if selected_field:
        # Convert selected field to shapely geometry
        field_geom = box(*selected_field['geometry']['coordinates'][0])
        # Get image chip for the selected field
        image_chip = get_image_chip(field_geom)
        # Run inference on the image chip
        crop_type = run_inference(image_chip)
        # Display the result
        st.write(f'Predicted Crop Type: {crop_type}')

        # Display the 224x224 image chip
        fig, ax = plt.subplots()
        ax.imshow(image_chip[0], cmap='gray')  # Assuming the first band for simplicity
        ax.set_title(f'Selected Field - {crop_type}')
        st.pyplot(fig)
