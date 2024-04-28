import streamlit as st
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

def extract_text_with_tesseract(pdf_path):
    """Use Tesseract to extract text from all pages of a PDF."""
    doc = fitz.open(pdf_path)
    text = ''
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_text = pytesseract.image_to_string(img)
        text += page_text
    doc.close()
    return text

def main():
    st.title("OCR Testing with Tesseract")
    pdf_path = 'docs/s50mcc.pdf'
    if st.button("Extract Text"):
        extracted_text = extract_text_with_tesseract(pdf_path)
        st.text_area("Extracted Text", extracted_text, height=300)

if __name__ == "__main__":
    main()
