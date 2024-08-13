import streamlit as st
from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim

# Define the bounding box
ZUERICH_BBOX = [8.364, 47.240, 9.0405, 47.69894]

def within_bbox(lat, lon, bbox):
    """Check if a point is within the given bounding box."""
    return bbox[1] <= lat <= bbox[3] and bbox[0] <= lon <= bbox[2]

def select_coordinates():
    st.title("Step 1: Select Location")

    instructions = """
    1. Choose a crop classification location. Search for a location or click on the map. 
    2. Proceed to the "Perform Crop Classification" step.
    
    _Note:_ The location must be within the green ZüriCrop area.
    """
    st.sidebar.header("Instructions")
    st.sidebar.markdown(instructions)

    # Initialize a map centered around the midpoint of the bounding box
    midpoint_lat = (ZUERICH_BBOX[1] + ZUERICH_BBOX[3]) / 2
    midpoint_lon = (ZUERICH_BBOX[0] + ZUERICH_BBOX[2]) / 2
    m = folium.Map(location=[midpoint_lat, midpoint_lon], zoom_start=9)

    # Add the bounding box to the map as a rectangle
    folium.Rectangle(
        bounds=[[ZUERICH_BBOX[1], ZUERICH_BBOX[0]], [ZUERICH_BBOX[3], ZUERICH_BBOX[2]]],
        color="green",
        fill=True,
        fill_opacity=0.1
    ).add_to(m)

    # Search for a location
    geolocator = Nominatim(user_agent="streamlit-app")
    location_query = st.text_input("Search for a location:")
    
    if location_query:
        location = geolocator.geocode(location_query)
        if location:
            lat, lon = location.latitude, location.longitude
            folium.Marker([lat, lon], tooltip=location.address).add_to(m)
            m.location = [lat, lon]
            m.zoom_start = 12
            
            if within_bbox(lat, lon, ZUERICH_BBOX):
                st.success(f"Location found: {location.address}. It is within the bounding box.")
                st.session_state["selected_location"] = (lat, lon)
            else:
                st.error(f"Location found: {location.address}. It is outside the bounding box.")
        else:
            st.error("Location not found. Please try again.")

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
    
    # Proceed to the next step
    link_disabled = "selected_location" not in st.session_state
    st.sidebar.page_link("pages/2_Perform_Crop_Classification.py", label="Proceed to Crop Classification", icon="🌾", disabled=link_disabled)

if __name__ == "__main__":
    select_coordinates()