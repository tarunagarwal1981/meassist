import streamlit as st
import pandas as pd
import openai
import os
from pathlib import Path

# Function to securely retrieve the OpenAI API key
def get_api_key():
    """Retrieve the API key from Streamlit secrets or environment variables."""
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

# Set the API key for OpenAI
openai.api_key = get_api_key()

# Function to load Excel data from a file
def load_excel_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Function to generate a textual summary of the DataFrame
def data_to_text(df):
    summary = "Data Summary:\n"
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            summary += f"{column}: Values range from {df[column].min()} to {df[column].max()}, with an average of {df[column].mean():.2f}.\n"
        else:
            unique_values = df[column].nunique()
            summary += f"{column}: Contains non-numeric data with {unique_values} unique entries.\n"
    return summary

# Function to query OpenAI using the ChatCompletion API
def query_llm(text_summary, user_query):
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text_summary},
        {"role": "user", "content": user_query}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        max_tokens=150
    )
    return response.choices[0].message['content']

def main():
    st.title("Data Analysis Tool")

    # Assuming the UOG folder is in the same directory as your script
    folder_path = Path("./UOG")
    files = list(folder_path.glob('*.xlsx'))
    
    if files:
        file_selector = st.selectbox('Select an Excel file', files)
        df = load_excel_data(file_selector)
        
        if not df.empty:
            text_summary = data_to_text(df)
            st.write(text_summary)  # Display the data summary
            user_query = st.text_input("Please enter your query about the data:")  # Get user's query
            
            if user_query and st.button("Generate Insights"):
                insights = query_llm(text_summary, user_query)
                st.write("Insights based on your query:", insights)
        else:
            st.error("The selected file is empty or invalid.")
    else:
        st.error("No Excel files found in the UOG directory.")

if __name__ == "__main__":
    main()
