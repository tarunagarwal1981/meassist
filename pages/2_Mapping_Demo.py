import streamlit as st
import os
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import faiss
import openai
from transformers import GPT2Tokenizer

# Define the path for the PDF document
pdf_path = 'docs/s50mcc.pdf'

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
        text = extract_text(pdf_path)
        if not text.strip():  # If text extraction fails, use OCR
            doc = fitz.open(pdf_path)
            text = ''.join(page.get_text() for page in doc)
            doc.close()
        return text
    except Exception as e:
        st.error(f"Error extracting text from {pdf_path}: {e}")
        return None

def process_single_pdf(pdf_path):
    """Processes a single PDF and extracts text."""
    text = extract_text_from_pdf(pdf_path)
    return [text] if text else []

def generate_embeddings(text_list):
    """Generates embeddings for a list of text documents."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(text_list, show_progress_bar=True)

def create_faiss_index(embeddings):
    """Creates a FAISS index for a set of document embeddings."""
    if embeddings is None or not isinstance(embeddings, np.ndarray):
        raise ValueError("Invalid or empty embeddings array.")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_documents(query, index, text_list, top_k=1):
    """Searches the index for the document most similar to the query."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [(text_list[i], distances[0][j]) for i, j in enumerate(indices[0])]

def main():
    st.title("Document Inquiry")
    document_text = process_single_pdf(pdf_path)
    if document_text:
        document_embeddings = generate_embeddings(document_text)
        faiss_index = create_faiss_index(np.array(document_embeddings))

        query = st.text_input("Enter your question:", "")
        if query:
            with st.spinner('Searching for answers...'):
                retrieved_docs = search_documents(query, faiss_index, document_text)
                if not retrieved_docs:
                    st.write("No relevant information could be found.")
                else:
                    answer = f"Based on the document, your question about '{query}' is answered as follows: {retrieved_docs[0][0]}"
                    st.write("Answer:", answer)

if __name__ == "__main__":
    main()
