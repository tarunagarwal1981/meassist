import streamlit as st
import os
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# Assuming pytesseract is properly configured with the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Update this path based on your server configuration

pdf_path = 'pages/s50mcc.pdf'  # Specify the path to your PDF here

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ''
        for page in doc:
            # Try to get text directly
            text += page.get_text()
        
        if not text.strip():  # If no text was extracted, attempt OCR with Tesseract
            for page in doc:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img)
        
        doc.close()
        return text, None
    except Exception as e:
        return None, str(e)

def main():
    st.title("Document Processor with OCR")
    if st.button("Process Document"):
        text, error = extract_text_from_pdf(pdf_path)
        if error:
            st.error(f"Error extracting text from {pdf_path}: {error}")
        if text:
            st.write("Text extracted from document:")
            st.text(text[:1000])  # Displaying only the first 1000 characters of the text

if __name__ == "__main__":
    main()
