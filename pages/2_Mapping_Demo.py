import streamlit as st
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def main():
    st.title("PDF OCR Testing")
    file_path = 'path/to/your/pdf/s50mcc.pdf'  # Update this to your PDF file path

    if st.button("Extract Text"):
        text = extract_text_from_pdf(file_path)
        # Display the first 1000 words
        if text:
            st.write(" ".join(text.split()[:1000]))  # Displays the first 1000 words
        else:
            st.write("No text could be extracted from the PDF.")

if __name__ == "__main__":
    main()
