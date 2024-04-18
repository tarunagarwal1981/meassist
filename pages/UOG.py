import streamlit as st
import pandas as pd
import os
import openai
from pathlib import Path

def get_api_key():
    """Retrieve the API key from Streamlit secrets or environment variables."""
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    return st.secrets.get('OPENAI_API_KEY', 'Your-OpenAI-API-Key')  # Replace 'Your-OpenAI-API-Key' with your actual key

# Set up the directory path
DIR_PATH = Path(__file__).parent.resolve() / "UOG"

# Load the Excel files from the directory
xlsx_files = []
for file_path in DIR_PATH.glob("*.xlsx"):
    xlsx_data = pd.read_excel(file_path)
    xlsx_files.append(xlsx_data)

# Set up the OpenAI Assistant API
openai.api_key = get_api_key()
openai.headers = {"OpenAI-Beta": "assistants=v2"}

# Create an Assistant
assistant = openai.beta.assistants.create(
    name="Excel Data Assistant",
    instructions="You are an assistant that can analyze and answer questions about Excel data.",
    model="gpt-4-turbo",
    tools=[{"type": "retrieval"}]
)

# Upload the Excel files to the Assistant
files = []
for xlsx_data in xlsx_files:
    file = openai.files.create(
        file=xlsx_data.to_csv(index=False),
        purpose="assistants",
    )
    files.append(file.id)

# Update the Assistant with the uploaded files
assistant = openai.beta.assistants.update(assistant.id, file_ids=files)

# Streamlit app
st.title("Excel Data Chat Assistant")

user_query = st.text_input("Ask a question about the data:")

if user_query:
    # Create a thread
    thread = openai.beta.threads.create()

    # Add the user query to the thread
    openai.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_query
    )

    # Run the Assistant on the thread
    run = openai.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # Wait for the run to complete
    while run.status != "completed":
        run = openai.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )

    # Get the Assistant's response
    messages = openai.beta.threads.messages.list(thread_id=thread.id)
    assistant_response = [m.content[0].text.value for m in messages if m.role == "assistant"]
    st.write(assistant_response[0])
