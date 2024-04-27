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
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# Define the folder path for document processing
folder_path = 'docs'

def get_api_key():
    """Retrieve the API key from Streamlit secrets or environment variables."""
    api_key = st.secrets['openai']['api_key'] if 'openai' in st.secrets else os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("API key not found. Set OPENAI_API_KEY as an environment variable.")
        st.stop()
    return api_key

openai.api_key = get_api_key()

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using pdfminer and optionally Doctr for OCR."""
    try:
        with open(pdf_path, 'rb') as file:
            text = extract_text(file)
        if not text.strip():
            model = ocr_predictor(pretrained=True)
            doc = DocumentFile.from_pdf(pdf_path)
            result = model(doc)
            text = ' '.join([block.text for page in result.pages for block in page.blocks])
        return text
    except Exception as e:
        st.error(f"Error extracting text from {pdf_path}: {e}")
        return None

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    try:
        with open(docx_path, 'rb') as file:
            doc = Document(file)
        return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        st.error(f"Error extracting text from {docx_path}: {e}")
        return None

def extract_text_from_excel(excel_path):
    """Extracts concatenated text from all sheets in an Excel file."""
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
                text = extract_text_from_pdf(file_path)
            elif file.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            elif file.endswith(('.xls', '.xlsx')):
                text = extract_text_from_excel(file_path)
            if text:
                document_texts.append(text)
                filenames.append(file)
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

def search_documents(query, index, text_list, top_k=5):
    """Searches the index for the documents most similar to the query."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [(text_list[idx], distances[0][i], idx) for i, idx in enumerate(indices[0])]

def create_augmented_prompt(query, retrieved_documents, filenames, top_k=3):
    """Creates an augmented prompt by combining the query with top retrieved documents."""
    instruction = "Give a detailed answer unless asked for a brief or concise or short answer."
    context = instruction

    for doc, _, idx in sorted(retrieved_documents, key=lambda x: x[1])[:top_k]:
        context += " " + doc

    return f"Based on the following information: {context}\n\nAnswer the question: {query}", [filenames[i] for _, _, i in retrieved_documents[:top_k]]

def generate_response(augmented_prompt, sources):
    """Generates a response using the OpenAI ChatCompletion API."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": augmented_prompt}
        ]
    )
    return response.choices[0].message['content'] + "\n\nSources: " + ", ".join(sources)

def main():
    st.title("Document Analysis Tool")
    document_texts, filenames = process_files_in_folder(folder_path)
    document_embeddings = generate_embeddings(document_texts)
    faiss_index = create_faiss_index(document_embeddings)

    query = st.text_input("Enter your question or type 'exit' to quit:", "")
    if query.lower() == 'exit':
        st.write("Exiting the program. Goodbye!")
        return

    with st.spinner('Searching documents...'):
        retrieved_docs = search_documents(query, faiss_index, document_texts, top_k=5)
        if not retrieved_docs:
            st.write("Sorry, no relevant information could be found for your question.")
            return
        augmented_prompt, sources = create_augmented_prompt(query, retrieved_docs, filenames, top_k=3)
        response = generate_response(augmented_prompt, sources)
        st.write("Answer:", response)

if __name__ == "__main__":
    main()
