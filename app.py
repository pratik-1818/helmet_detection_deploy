import streamlit as st

# Title of the Streamlit app
st.title("Helmet Detection System")

# Display video feed from FastAPI
st.image("http://127.0.0.1:8000/video_feed/", caption="Real-Time Detection Stream", use_container_width=True)
