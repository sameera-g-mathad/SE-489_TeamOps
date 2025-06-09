# app.py - Your first Streamlit app
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




# Set page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="My First Streamlit App",
    page_icon="🎉",
    layout="wide",
    initial_sidebar_state="expanded"
)
# # Success message
# st.success("✅ Your Streamlit app is working!")

# Main title
st.title("🎉 Welcome to Streamlit!")

# Subtitle
st.header("This is my first Streamlit application")

text = st.text_area("Enter your text here:", placeholder='Type something...', height=100)


def submit_request():
    """
    Function to handle the submit button click.
    It processes the text input and displays a message.
    """
    st.write("You submitted:")
    st.write(text)
    st.success("✅ Your input has been processed!")

st.button('Submit', on_click=submit_request)