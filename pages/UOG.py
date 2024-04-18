import streamlit as st
import pandas as pd
import os
import openai
from pathlib import Path
import tempfile

def get_api_key():
    """Retrieve the API key from Streamlit secrets or environment variables."""
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    return os.getenv('OPENAI_API_KEY', 'Your-OpenAI-API-Key')

# Set up the directory path
DIR_PATH = Path(__file__).parent.resolve() / "UOG"

# Load the Excel files from the directory
xlsx_files = []
for file_path in DIR_PATH.glob("*.xlsx"):
    xlsx_data = pd.read_excel(file_path)
    xlsx_files.append(xlsx_data)

# Set up the OpenAI Assistant API
openai.api_key = get_api_key()

# Create an Assistant
assistant = openai.Assistant.create(
    name="Excel Data Assistant",
    instructions="You are an assistant that can analyze and answer questions about Excel data.",
    model="gpt-4-turbo-preview",
    tools=[{"type": "file_search"}]
)

# Upload the Excel files to the Assistant
files = []
for xlsx_data in xlsx_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmpfile:
        xlsx_data.to_csv(tmpfile.name, index=False)
        file = openai.File.create(
            file=open(tmpfile.name, 'rb'),
            purpose="assistants",
        )
        files.append(file.id)

# Update the Assistant with the uploaded files
assistant = openai.Assistant.update(assistant.id, file_ids=files)

# Streamlit app
st.title("Excel Data Chat Assistant")

user_query = st.text_input("Ask a question about the data:")

if user_query:
    # Create a thread
    thread = openai.Thread.create()

    # Add the user query to the thread
    openai.Message.create(
        thread_id=thread.id,
        role="user",
        content=user_query
    )

    # Run the Assistant on the thread
    run = openai.Run.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # Wait for the run to complete
    while run.status != "completed":
        run = openai.Run.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )

    # Get the Assistant's response
    messages = openai.Message.list(thread_id=thread.id)
    assistant_response = [m.content for m in messages.data if m.role == "assistant"]
    st.write(assistant_response[0].content if assistant_response else "No response from assistant.")
