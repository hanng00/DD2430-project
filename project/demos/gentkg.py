import streamlit as st

st.title("TKG Visualizer")

# Define options for dropdowns (These can be dynamic or fetched from an API in real scenarios)
subjects = ["Subject 1", "Subject 2", "Subject 3"]
relations = ["Relation A", "Relation B", "Relation C"]
timesteps = list(range(1, 11))  # For example, timesteps from 1 to 10

# Create dropdowns
selected_subject = st.selectbox("Select a Subject", subjects)
selected_relation = st.selectbox("Select a Relation", relations)
selected_timestep = st.selectbox("Select a Timestep", timesteps)
