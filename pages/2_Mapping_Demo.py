import streamlit as st
import pandas as pd
import spacy
from spacy.tokens import Span
from spacy.language import Language
from spacy.matcher import PhraseMatcher
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
import numpy as np
import dataframe_image as dfi
from docx import Document
from docx.shared import Inches
from textwrap import wrap
from io import BytesIO
import openai
import os

# Function to get the OpenAI API key
def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    return os.getenv('OPENAI_API_KEY', 'Your-OpenAI-API-Key')

openai.api_key = get_api_key()  # Set the OpenAI API key

# Download spaCy model if not already present
spacy_model = "en_core_web_sm"
try:
    nlp = spacy.load(spacy_model)
except OSError:
    from spacy.cli import download
    download(spacy_model)
    nlp = spacy.load(spacy_model)

# Streamlit app
def main():
    st.title("Vessel Performance Report Generator")

    hull_file_url = 'https://drive.google.com/uc?export=download&id=1dNjLhrvUIWuwiQh9QIZ-7Xc0ba3ze23m'
    performance_file_url = 'https://drive.google.com/uc?export=download&id=1dK9C9niJAm040YCOij-UxziBZIfkJTij'

    st.write("Debug: Streamlit app is running.")

    vessel_names = st.text_input("Enter vessel names (comma-separated)")
    st.write(f"Debug: Vessel names - {vessel_names}")

    query = st.text_input("Enter your query (e.g., 'hull performance', 'vessel performance')")
    st.write(f"Debug: Query - {query}")

    if st.button("Generate Report"):
        st.write("Debug: Button clicked.")
        if hull_file_url and performance_file_url and vessel_names and query:
            st.write("Debug: Processing inputs.")

if __name__ == "__main__":
    main()
