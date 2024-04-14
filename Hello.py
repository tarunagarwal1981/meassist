import os
import numpy as np
import pandas as pd
from pdfminer.high_level import extract_text
from docx import Document
import openpyxl
from sentence_transformers import SentenceTransformer
import faiss
import openai

# Define the folder path for document processing
folder_path = 'docs'

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        return extract_text(pdf_path)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    try:
        doc = Document(docx_path)
        return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        print(f"Error extracting text from {docx_path}: {e}")
        return None

def extract_text_from_excel(excel_path):
    """Extracts text from an Excel file."""
    try:
        df = pd.concat(pd.read_excel(excel_path, sheet_name=None), ignore_index=True)
        return df.to_csv(index=False, header=False)
    except Exception as e:
        print(f"Error extracting text from {excel_path}: {e}")
        return None

def process_files_in_folder(folder_path):
    """Processes files in a specified folder and extracts texts."""
    document_texts = []
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
    return document_texts

def generate_embeddings(text_list):
    """Generates embeddings for a list of text documents."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_list, show_progress_bar=True)
    return embeddings

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
    return [(text_list[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

def create_augmented_prompt(query, retrieved_documents, top_k=3):
    """Creates an augmented prompt by combining the query with top retrieved documents."""
    if len(retrieved_documents) < top_k:
        top_k = len(retrieved_documents)
    context = " ".join([doc[0] for doc in sorted(retrieved_documents, key=lambda x: x[1])[:top_k]])
    return f"Based on the following information: {context}\n\nAnswer the question: {query}"

def generate_response_with_gpt(augmented_prompt):
    """Generates a response using the OpenAI ChatCompletion API."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": augmented_prompt}]
    )
    return response.choices[0].message['content']

def main():
    """Main function to handle user queries interactively."""
    document_texts = process_files_in_folder(folder_path)
    document_embeddings = generate_embeddings(document_texts)
    faiss_index = create_faiss_index(np.array(document_embeddings))

    print("Please enter your question or type 'exit' to quit:")
    while True:
        query = input("Your question: ")
        if query.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break

        retrieved_docs = search_documents(query, faiss_index, document_texts)
        if not retrieved_docs:
            print("Sorry, no relevant information could be found for your question.")
            continue

        augmented_prompt = create_augmented_prompt(query, retrieved_docs)
        response = generate_response_with_gpt(augmented_prompt)
        print("Answer:", response)

if __name__ == "__main__":
    main()
