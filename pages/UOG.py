import streamlit as st
import pandas as pd
from pathlib import Path
import openai
import json

# Function to securely retrieve the OpenAI API key
def get_api_key():
    """Retrieve the API key from Streamlit secrets or environment variables."""
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    return st.secrets.get('OPENAI_API_KEY', 'Your-OpenAI-API-Key')  # Replace 'Your-OpenAI-API-Key' with your actual key

openai.api_key = get_api_key()

def load_excel_data(file_path):
    """Load Excel data from a file."""
    df = pd.read_excel(file_path)
    return df

def data_to_text(df):
    """Generate a compact summary of the DataFrame for the LLM."""
    summary = f"Total rows: {len(df)}\n"
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            summary += f"{column}: Avg={df[column].mean():.2f}, Min={df[column].min()}, Max={df[column].max()}\n"
        else:
            unique_values = df[column].nunique()
            summary += f"{column}: {unique_values} unique values\n"
    return summary

def query_llm_conversation(messages):
    """Query GPT-4 using ChatCompletion for a conversational response based on messages."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=500
    )
    return response.choices[0].message['content']

def main():
    st.title("Data Analysis with AI Assistant")

    # Load data
    folder_path = Path("pages/UOG")
    files = list(folder_path.glob('*.xlsx'))

    if files:
        file_selector = st.selectbox('Select an Excel file:', files)
        df = load_excel_data(file_selector)
        
        if not df.empty:
            # Display DataFrame
            st.dataframe(df.head())
            user_query = st.text_input("Enter your query or request for the assistant:")

            if user_query:
                # Pre-process data and prepare messages for chat
                data_summary = data_to_text(df)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": data_summary},
                    {"role": "user", "content": user_query}
                ]
                
                if st.button("Get Insights"):
                    insights = query_llm_conversation(messages)
                    st.write("Assistant Response:", insights)
        else:
            st.error("The selected file is empty or invalid.")
    else:
        st.error("No Excel files found in the directory.")

if __name__ == "__main__":
    main()
