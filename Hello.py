
# Install necessary libraries

# Depending on your specific needs, you might need to install additional packages or specific versions of these packages.

# Basic imports for data handling
import pandas as pd
import numpy as np

# Library imports for file handling
from pdfminer.high_level import extract_text
from docx import Document
import openpyxl

# Imports for embeddings and FAISS
from sentence_transformers import SentenceTransformer
import faiss


# LangChain or other specific libraries can be imported based on your usage
# import langchain  # Uncomment or modify based on your project's requirements

# Replace this line
# folder_path = '/content/drive/My Drive/LLM'

# With this line
import os
folder_path = os.path.join(os.getcwd(), 'docs')

import os
from pdfminer.high_level import extract_text
from docx import Document
import pandas as pd

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    try:
        doc = Document(docx_path)
        fullText = [paragraph.text for paragraph in doc.paragraphs]
        return '\n'.join(fullText)
    except Exception as e:
        print(f"Error extracting text from {docx_path}: {e}")
        return None

def extract_text_from_excel(excel_path):
    """Extracts text from an Excel file, concatenating all text from the workbook."""
    try:
        df = pd.concat(pd.read_excel(excel_path, sheet_name=None), ignore_index=True)
        text = df.to_csv(index=False, header=False)
        return text
    except Exception as e:
        print(f"Error extracting text from {excel_path}: {e}")
        return None

def process_files_in_folder(folder_path):
    document_texts = []  # Initialize a list to hold document texts
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif file.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            elif file.endswith(('.xls', '.xlsx')):
                text = extract_text_from_excel(file_path)
            else:
                text = None

            if text:
                document_texts.append(text)  # Store the extracted text
    return document_texts


# Replace 'folder_path' with your folder's path
folder_path = os.path.join(os.getcwd(), 'docs')  # Adjust this path
process_files_in_folder(folder_path)

from sentence_transformers import SentenceTransformer

# Initialize a pre-trained model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(text_list):
    """Generates embeddings for a list of text documents."""
    embeddings = model.encode(text_list, show_progress_bar=True)
    return embeddings

import faiss
import numpy as np

def create_faiss_index(embeddings):
    """Creates a FAISS index for a set of document embeddings."""
    dimension = embeddings.shape[1]  # Get the dimension of embeddings
    index = faiss.IndexFlatL2(dimension)  # Use the FlatL2 index for Euclidean distance
    index.add(embeddings)  # Add embeddings to the index
    return index

# Example usage:
# Assume `document_embeddings` is your array of document embeddings from the previous step
# document_embeddings = generate_embeddings(your_document_texts)
# faiss_index = create_faiss_index(np.array(document_embeddings))

def search_documents(query, index, text_list, top_k=5):
    """Searches the index for the documents most similar to the query."""
    query_embedding = model.encode([query])[0]  # Convert query to embedding
    distances, indices = index.search(np.array([query_embedding]), top_k)  # Find the top_k closest embeddings

    results = [(text_list[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return results

# Example usage:
# results = search_documents("Your query here", faiss_index, your_document_texts)
# for text, score in results:
#     print(f"Score: {score:.2f}, Text: {text[:200]}...")  # Print the beginning of each matching document


import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response_with_gpt(augmented_prompt):
    """
    Generates a response using the OpenAI ChatCompletion API for chat models.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": augmented_prompt}
        ]
    )
    # Extracting the text response from the list of messages
    # The response structure might need adjustment based on the actual output
    return response.choices[0].message['content']

def create_augmented_prompt(query, retrieved_documents, top_k=3):
    """Creates an augmented prompt by combining the query with top retrieved documents."""
    # Sort retrieved documents by their score (assuming the second tuple element is the score)
    retrieved_documents = sorted(retrieved_documents, key=lambda x: x[1], reverse=True)

    # Select the top_k documents and concatenate their text
    context = " ".join([doc[0] for doc in retrieved_documents[:top_k]])

    # Combine the original query with the context
    augmented_prompt = f"Based on the following information: {context} \n\nAnswer the question: {query}"
    return augmented_prompt

def handle_query(query):
    """Handles a user query by retrieving relevant documents and generating a response."""
    # Step 1: Retrieve relevant documents
    retrieved_docs = search_documents(query, faiss_index, your_document_texts)

    # Step 2: Create an augmented prompt with the query and retrieved documents
    augmented_prompt = create_augmented_prompt(query, retrieved_docs)

    # Step 3: Generate a response using GPT with the augmented prompt
    response = generate_response_with_gpt(augmented_prompt)

    return response

# Example usage:
# response = handle_query("What are the main benefits of product X?")
# print(response)

# Replace 'folder_path' with your folder's path
folder_path = '/content/drive/My Drive/LLM'  # Adjust this path
your_document_texts = process_files_in_folder(folder_path)  # Process documents and collect texts

# Generate embeddings for these texts
document_embeddings = generate_embeddings(your_document_texts)

# Create a FAISS index with these embeddings
# First, ensure embeddings is a NumPy array for FAISS
document_embeddings_np = np.asarray(document_embeddings)
faiss_index = create_faiss_index(document_embeddings_np)

# Make sure to define or import all the necessary functions and initialize APIs and models before this snippet.

def main():
    print("Please enter your question or 'exit' to quit:")
    while True:
        query = input("Your question: ")
        if query.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break

        # Retrieve relevant documents for the query
        retrieved_docs = search_documents(query, faiss_index, your_document_texts)

        if not retrieved_docs:
            print("Sorry, no relevant information could be found for your question.")
            continue

        # Create an augmented prompt combining the query with the retrieved documents
        augmented_prompt = create_augmented_prompt(query, retrieved_docs)

        # Generate a response using GPT based on the augmented prompt
        response = generate_response_with_gpt(augmented_prompt)

        print("Answer:", response)

if __name__ == "__main__":
    main()
