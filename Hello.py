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

def process_files_in_folder(folder_path, retrain=False):
    """Processes files in a specified folder and extracts texts."""
    if retrain or 'document_texts' not in st.session_state:
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
                    filenames.append(file)
        st.session_state['document_texts'] = document_texts
        st.session_state['filenames'] = filenames
    return st.session_state['document_texts'], st.session_state['filenames']

def generate_embeddings(text_list, retrain=False):
    """Generates embeddings for a list of text documents."""
    if retrain or 'embeddings' not in st.session_state:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(text_list, show_progress_bar=True)
        st.session_state['embeddings'] = embeddings
    return st.session_state['embeddings']

def create_faiss_index(embeddings, retrain=False):
    """Creates a FAISS index for a set of document embeddings."""
    if retrain or 'faiss_index' not in st.session_state:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        st.session_state['faiss_index'] = index
    return st.session_state['faiss_index']

def main():
    st.title("Engine Expert")

    if st.button("Retrain Model"):
        document_texts, filenames = process_files_in_folder(folder_path, retrain=True)
        embeddings = generate_embeddings(document_texts, retrain=True)
        faiss_index = create_faiss_index(embeddings, retrain=True)
        st.success("Model retrained with new data!")

    document_texts, filenames = process_files_in_folder(folder_path)
    embeddings = generate_embeddings(document_texts)
    faiss_index = create_faiss_index(embeddings)

    query = st.text_input("Enter your question or type 'exit' to quit:", "")
    if query:
        if query.lower() == 'exit':
            st.write("Exiting the program. Goodbye!")
            st.stop()
        
        distances, indices = faiss_index.search(np.array([query_embedding]), top_k)
        results = [(document_texts[idx], distances[0][i], idx) for i, idx in enumerate(indices[0])]
        if not results:
            st.write("Sorry, no relevant information could be found for your question.")
        else:
            augmented_prompt, sources = create_augmented_prompt(query, results, filenames)
            response = generate_response_with_gpt(augmented_prompt, sources)
            st.write("Answer:", response)

if __name__ == "__main__":
    main()
