import streamlit as st
from streamlit_folium import st_folium
import folium

# Define the bounding box
ZUERICH_BBOX = [8.364, 47.240, 9.0405, 47.69894]

def within_bbox(lat, lon, bbox):
    """Check if a point is within the given bounding box."""
    return bbox[1] <= lat <= bbox[3] and bbox[0] <= lon <= bbox[2]

def select_coordinates():
    st.title("Step 1: Select POI Location")
    
    # Initialize a map centered around the midpoint of the bounding box
    midpoint_lat = (ZUERICH_BBOX[1] + ZUERICH_BBOX[3]) / 2
    midpoint_lon = (ZUERICH_BBOX[0] + ZUERICH_BBOX[2]) / 2
    m = folium.Map(location=[midpoint_lat, midpoint_lon], zoom_start=10)

    # Add the bounding box to the map as a rectangle
    folium.Rectangle(
        bounds=[[ZUERICH_BBOX[1], ZUERICH_BBOX[0]], [ZUERICH_BBOX[3], ZUERICH_BBOX[2]]],
        color="green",
        fill=False,
        fill_opacity=0.2,
        popup="Bounding Box"
    ).add_to(m)

    # Add a click event listener to capture coordinates
    m.add_child(folium.LatLngPopup())

    # Display the map using streamlit-folium
    st_data = st_folium(m, height=500, width=800)

    # Check if the user clicked within the bounding box
    if st_data["last_clicked"]:
        lat, lon = st_data["last_clicked"]["lat"], st_data["last_clicked"]["lng"]
        if within_bbox(lat, lon, ZUERICH_BBOX):
            st.success(f"Selected Location: Latitude {lat}, Longitude {lon}")
            st.session_state["selected_location"] = (lat, lon)
        else:
            st.error(f"Selected Location is outside the allowed area. Please select a location within the bounding box.")

if __name__ == "__main__":
    select_coordinates()
