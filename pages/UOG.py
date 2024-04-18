import streamlit as st
import pandas as pd
import os
import openai
from pathlib import Path
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

def get_api_key():
    """Retrieve the API key from Streamlit secrets or environment variables."""
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    return st.secrets.get('OPENAI_API_KEY', 'Your-OpenAI-API-Key') # Replace 'Your-OpenAI-API-Key' with your actual key

# Set up the directory path
DIR_PATH = Path(__file__).parent.resolve() / "UOG"

# Load the Excel files from the directory
xlsx_files = []
for file_path in DIR_PATH.glob("*.xlsx"):
    xlsx_data = pd.read_excel(file_path)
    xlsx_files.append(xlsx_data)

# Set up the OpenAI API
openai.api_key = get_api_key()
llm = OpenAI(api_token=openai.api_key)

# Combine the Excel data into a single DataFrame
combined_data = pd.concat(xlsx_files, ignore_index=True)

# Create a SmartDataframe object
smart_df = SmartDataframe(combined_data, config={"llm": llm})

# Streamlit app
st.title("Defect Sheet Chat Assistant")

user_query = st.text_input("Ask a question about the defect sheet data:")

if user_query:
    # Use PandasAI to answer the user query
    answer = smart_df.chat(user_query)
    st.write(answer)
    
