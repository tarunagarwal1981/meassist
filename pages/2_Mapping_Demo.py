import streamlit as st
import os
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
import pdf2image
import openai  # Import the OpenAI library

def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    return os.getenv('OPENAI_API_KEY', 'Your-OpenAI-API-Key')

openai.api_key = get_api_key()  # Set the OpenAI API key

def extract_text_from_pdf(pdf_path):
    try:
        # Convert PDF to images
        images = pdf2image.convert_from_path(pdf_path)
        text = ''
        for img in images:
            text += pytesseract.image_to_string(img)
        return text, None
    except Exception as e:
        return None, str(e)

def process_files_in_folder(folder_path):
    document_texts = []
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.pdf'):
                text, error = extract_text_from_pdf(file_path)
                if error:
                    st.error(f"Error extracting text from {file_path}: {error}")
                if text:
                    document_texts.append(text)
                    filenames.append(file)
    return document_texts, filenames

def main():
    st.title("Document Processor with OCR")
    folder_path = 'pages'  # Define the folder path here
    pdf_path = 'pages/s50mcc.pdf'  # Specify the path to your PDF here
    if st.button("Process Documents"):
        document_texts, filenames = process_files_in_folder(folder_path)
        if document_texts:
            st.write("Texts extracted from documents:")
            for text in document_texts[:5]:  # Displaying only first 5 entries
                st.text(text)

if __name__ == "__main__":
    main()
