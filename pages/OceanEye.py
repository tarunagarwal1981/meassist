# app.py
import pandas as pd
import openai
import re
import os
import streamlit as st

# Path to the file in docs folder
file_path = 'docs/Hull Performance data.csv'
df = pd.read_csv(file_path)

def get_api_key():
    """Retrieve the API key from Streamlit secrets or environment variables."""
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

# Set your OpenAI API key
openai.api_key = get_api_key()

# Function to query the DataFrame based on natural language input
def query_dataframe(df, natural_language_query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that converts natural language queries into pandas DataFrame operations."},
            {"role": "user", "content": f"DataFrame columns: {list(df.columns)}"},
            {"role": "user", "content": f"Convert the following natural language query into a pandas DataFrame operation: {natural_language_query}"}
        ]
    )

    operation = response['choices'][0]['message']['content'].strip()
    st.write(f"Generated operation:\n{operation}")

    # Extract the code part from the response
    code_match = re.search(r"```python(.*?)```", operation, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        code = operation

    # Add safe checks and execute the code
    try:
        # We need to define the local variables so that eval can work correctly
        local_vars = {"df": df}
        
        # Execute the operation safely
        exec(f"result = {code}", {}, local_vars)
        result = local_vars['result']
        
        # Check if the result is a single value or DataFrame
        if isinstance(result, pd.DataFrame) and result.empty:
            st.write("The resulting DataFrame is empty.")
        elif isinstance(result, pd.Series) and result.empty:
            st.write("The resulting Series is empty.")
        else:
            return result
    except Exception as e:
        st.write(f"Error in evaluating operation: {e}")
        return None

# Streamlit app
st.title('Hull Performance Data Query App')

natural_language_query = st.text_input("Please enter your query:")
if natural_language_query:
    result = query_dataframe(df, natural_language_query)
    if result is not None:
        st.write("Query result:")
        st.write(result)
    else:
        st.write("Could not process the query.")
