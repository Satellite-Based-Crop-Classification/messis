import streamlit as st

def main():
    st.set_page_config(layout="wide", page_title="Messis 🌾 - Crop Classification 🌎")

    st.title("Messis 🌾 - Crop Classification 🌎")

    st.write("Welcome to the Messis Crop Classification app. Use the sidebar to navigate between selecting coordinates and performing inference.")

    st.page_link("Home.py", label="Home", icon="🏠")
    st.page_link("pages/1_Select_Location.py", label="Select Location", icon="📍")
    st.page_link("pages/2_Perform_Inference.py", label="Perform Inference", icon="🔍")

if __name__ == "__main__":
    main()
