import streamlit as st
import pandas as pd
import os
import openai
import gzip
import base64
from pathlib import Path

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

# Set up the OpenAI Assistant API
openai.api_key = get_api_key()
openai.headers = {"OpenAI-Beta": "assistants=v2"}

# Combine the Excel data into a single DataFrame
combined_data = pd.concat(xlsx_files, ignore_index=True)

# Convert the combined data to CSV format
csv_data = combined_data.to_csv(index=False)

# Compress the CSV data with a higher compression level
compressed_csv_data = gzip.compress(csv_data.encode('utf-8'), compresslevel=9)

# Encode the compressed data as base64
encoded_csv_data = base64.b64encode(compressed_csv_data).decode('utf-8')

# Split the encoded CSV data into chunks
chunk_size = 200000  # Adjust the chunk size as needed
data_chunks = [encoded_csv_data[i:i+chunk_size] for i in range(0, len(encoded_csv_data), chunk_size)]

# Create an Assistant
assistant = openai.beta.assistants.create(
    name="Defect Sheet Assistant",
    instructions="You are an assistant that can analyze and answer questions about defect sheet data of a fleet of ships.",
    model="gpt-4-turbo-preview",
    tools=[{"type": "file_search"}]
)

# Streamlit app
st.title("Defect Sheet Chat Assistant")

user_query = st.text_input("Ask a question about the defect sheet data:")

if user_query:
    # Create a thread
    thread = openai.beta.threads.create()

    # Add each chunk as a separate message to the thread
    for i, chunk in enumerate(data_chunks):
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Data chunk {i+1}/{len(data_chunks)}:\n\n{chunk}\n\n"
                    f"This is part of the compressed CSV data containing information about defects in a fleet of ships."
        )

    # Add the user query as a message to the thread
    openai.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"Question: {user_query}\n"
                f"The data contains information about defects in a fleet of ships, including columns for vessel name, defect name, "
                f"equipment/component, subcomponent, status (open/closed), expected budget spending, action taken or planned action, "
                f"progress, and other relevant information.\n"
                f"Please provide a specific and accurate answer based on the defect sheet data provided across all the data chunks."
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

    # Post-process the Assistant's response
    processed_response = assistant_response[0]
    # Add any additional post-processing steps here if needed

    st.write(processed_response)
