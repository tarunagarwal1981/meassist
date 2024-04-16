import streamlit as st
import os
import numpy as np
import pandas as pd
from pdfminer.high_level import extract_text
from docx import Document
import openpyxl
from sentence_transformers import SentenceTransformer
import faiss
import openai
from transformers import GPT2Tokenizer

# Define the folder path for document processing
folder_path = 'docs'

def get_api_key():
    """Retrieve the API key from Streamlit secrets or environment variables."""
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

openai.api_key = get_api_key()

if 'history' not in st.session_state:
    st.session_state['history'] = []

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        return extract_text(pdf_path)
    except Exception as e:
        st.error(f"Error extracting text from {pdf_path}: {e}")
        return None

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    try:
        doc = Document(docx_path)
        return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        st.error(f"Error extracting text from {docx_path}: {e}")
        return None

def extract_text_from_excel(excel_path):
    """Extracts text from an Excel file."""
    try:
        df = pd.concat(pd.read_excel(excel_path, sheet_name=None), ignore_index=True)
        return df.to_csv(index=False, header=False)
    except Exception as e:
        st.error(f"Error extracting text from {excel_path}: {e}")
        return None

@st.cache(allow_output_mutation=True, show_spinner=False)
def process_files_in_folder(folder_path):
    """Processes files in a specified folder and extracts texts."""
    document_texts = []
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            text = None
            if file.endswith('.pdf'):
                text = extract_text_from_pdf(open(file_path, 'rb'))
            elif file.endswith('.docx'):
                text = extract_text_from_docx(open(file_path, 'rb'))
            elif file.endswith(('.xls', '.xlsx')):
                text = extract_text_from_excel(open(file_path, 'rb'))
            if text:
                document_texts.append(text)
                filenames.append(file)  # Keep track of file names for source attribution
    return document_texts, filenames

@st.cache(allow_output_mutation=True, show_spinner=False)
def generate_embeddings(text_list):
    """Generates embeddings for a list of text documents."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_list, show_progress_bar=True)
    return embeddings

@st.cache(allow_output_mutation=True, show_spinner=False)
def create_faiss_index(embeddings):
    """Creates a FAISS index for a set of document embeddings."""
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def send_query(query):
    """Send a query to the OpenAI API using the accumulated chat history."""
    conversation_history = st.session_state['history']
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": query}]
    )
    answer = response.choices[0].message['content']
    conversation_history.append(f"User: {query}")
    conversation_history.append(f"Assistant: {answer}")
    return answer

def main():
    st.title("Engine Expert")
    if st.button("Start New Chat"):
        st.session_state['history'] = []  # Reset conversation history
        st.success("New chat started. Previous context cleared.")

    query = st.text_input("Enter your question or type 'exit' to quit:", "")
    if query:
        if query.lower() == 'exit':
            st.write("Exiting the program. Goodbye!")
            st.stop()

        document_texts, filenames = process_files_in_folder(folder_path)
        document_embeddings = generate_embeddings(document_texts)
        faiss_index = create_faiss_index(np.array(document_embeddings))

        if not st.session_state['history']:
            # Populate the initial conversation context if starting new
            st.session_state['history'].extend([
                f"System: Initializing document analysis with context from {len(document_texts)} documents."
            ])

        with st.spinner('Processing your query...'):
            response = send_query(query)
            st.write("Answer:", response)

if __name__ == "__main__":
    main()
