import streamlit as st
import pandas as pd
from pathlib import Path
import openai

# Function to securely retrieve the OpenAI API key
def get_api_key():
    """Retrieve the API key from Streamlit secrets or environment variables."""
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    return st.secrets.get('OPENAI_API_KEY', 'Your-OpenAI-API-Key')  # Replace 'Your-OpenAI-API-Key' with your actual key

openai.api_key = get_api_key()

def data_to_text(df):
    """Generate a textual summary of the DataFrame."""
    summary = "Data Summary:\n"
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            summary += f"{column}: Values range from {df[column].min()} to {df[column].max()}, and the average is {df[column].mean():.2f}.\n"
        else:
            unique_values = df[column].nunique()
            summary += f"{column}: Contains non-numeric data with {unique_values} unique entries.\n"
    return summary

def query_assistant(text_summary, user_query):
    """Query the GPT-4 Assistant API, integrating a code interpreter tool."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant equipped with a code interpreter."},
                {"role": "user", "content": text_summary},
                {"role": "user", "content": user_query}
            ],
            tools=[{"type": "code_interpreter"}],
            max_tokens=250
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    st.title("Data Analysis with AI Assistant")

    folder_path = Path("pages/UOG")
    files = list(folder_path.glob('*.xlsx'))

    if files:
        file_selector = st.selectbox('Select an Excel file:', files)
        df = pd.read_excel(file_selector)
        
        if not df.empty:
            text_summary = data_to_text(df)
            st.write(text_summary)
            user_query = st.text_input("Enter your query or request for the assistant:")

            if user_query and st.button("Get Insights"):
                insights = query_assistant(text_summary, user_query)
                st.write("Assistant Response:", insights)
        else:
            st.error("The selected file is empty or invalid.")
    else:
        st.error("No Excel files found in the directory.")

if __name__ == "__main__":
    main()
