
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
import fitz  # PyMuPDF

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
        text = extract_text(pdf_path)
        if not text.strip():  # Check if extracted text is empty
            doc = fitz.open(pdf_path)
            text = ''
            for page in doc:
                text += page.get_text()
            doc.close()
        return text
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
                text = extract_text_from_pdf(file_path)
            elif file.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            elif file.endswith(('.xls', '.xlsx')):
                text = extract_text_from_excel(file_path)
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
    if embeddings is None or not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("Invalid or empty embeddings array.")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_documents(query, index, text_list, top_k=5):
    """Searches the index for the documents most similar to the query."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [(text_list[idx], distances[0][i], idx) for i, idx in enumerate(indices[0])]

def create_augmented_prompt(query, retrieved_documents, filenames, top_k=3, max_tokens=16384):
    """Creates an augmented prompt by combining the query with top retrieved documents."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompt_tokens = tokenizer.encode(query)
    available_tokens = max_tokens - len(prompt_tokens) - 50  # Reserve some tokens for query and other text
    instruction = "Give a detailed answer unless asked for a brief or concise or short answer."
    context = instruction

    for doc, _, idx in sorted(retrieved_documents, key=lambda x: x[1])[:top_k]:
        doc_tokens = tokenizer.encode(doc)
        if len(doc_tokens) < available_tokens:
            context += " " + doc
            prompt_tokens += doc_tokens
            available_tokens = max_tokens - len(prompt_tokens)
        if available_tokens <= 0:
            break

    return f"Based on the following information: {context}\n\nAnswer the question: {query}", [filenames[i] for _, _, i in retrieved_documents[:top_k]]

def generate_response_with_gpt(augmented_prompt, sources, temperature=0.1, max_tokens=500, top_p=1.0):
    """Generates a response using the OpenAI ChatCompletion API with additional parameters."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": augmented_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    return response.choices[0].message['content'] + "\n\nSources: " + ", ".join(sources)

def main():
    st.title("Engine Expert")
    document_texts, filenames = process_files_in_folder(folder_path)
    document_embeddings = generate_embeddings(document_texts)
    faiss_index = create_faiss_index(np.array(document_embeddings))

    query = st.text_input("Enter your question or type 'exit' to quit:", "")
    if query:
        if query.lower() == 'exit':
            st.write("Exiting the program. Goodbye!")
            st.stop()

        with st.spinner('Processing your query...'):
            retrieved_docs = search_documents(query, faiss_index, document_texts)
            if not retrieved_docs:
                st.write("Sorry, no relevant information could be found for your question.")
            else:
                augmented_prompt, sources = create_augmented_prompt(query, retrieved_docs, filenames)
                response = generate_response_with_gpt(augmented_prompt, sources)
                st.write("Answer:", response)

if __name__ == "__main__":
    main()
