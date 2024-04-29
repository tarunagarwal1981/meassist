import os
import streamlit as st
import pandas as pd
from pdfminer.high_level import extract_text

# Define the folder path for document processing
folder_path = 'pages/s50mcc.pdf'

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using pdfminer.six."""
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        st.error(f"Error extracting text from {pdf_path}: {e}")
        return None

def main():
    st.title("PDF Text Extraction")
    if st.button("Extract Text"):
        text = extract_text_from_pdf(folder_path)
        if text:
            st.text_area("Extracted Text", text, height=300)
        else:
            st.write("Failed to extract text.")

if __name__ == "__main__":
    main()
