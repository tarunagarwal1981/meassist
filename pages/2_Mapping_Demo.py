import streamlit as st
import os
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    return os.getenv('OPENAI_API_KEY', 'Your-OpenAI-API-Key')

openai.api_key = get_api_key()

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ''
        for page in doc:
            # Try to get text directly
            text += page.get_text()
        
        if not text.strip():  # If no text was extracted
            # Attempt OCR with Tesseract
            for page in doc:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img)
        
        doc.close()
        return text
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
    if st.button("Process Documents"):
        document_texts, filenames = process_files_in_folder(folder_path)
        if document_texts:
            st.write("Texts extracted from documents:")
            for text in document_texts[:5]:  # Displaying only first 5 entries
                st.text(text)

if __name__ == "__main__":
    main()
