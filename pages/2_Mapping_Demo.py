import easyocr
from pdf2image import convert_from_path
import pytesseract
import random

def extract_random_text_from_pdf(pdf_path, num_chars=1000):
    # Convert PDF to images
    pages = convert_from_path(pdf_path)
    
    # Initialize easyocr reader
    reader = easyocr.Reader(['en'])  # Specify the language(s) for OCR
    
    # Extract text from all pages
    text = ''
    for page in pages:
        # Extract text from the image using pytesseract
        page_text = pytesseract.image_to_string(page)
        
        # Append the page text to the overall text
        text += page_text + '\n'
    
    # Extract a random substring of specified length from the text
    if len(text) > num_chars:
        start_index = random.randint(0, len(text) - num_chars)
        random_text = text[start_index:start_index + num_chars]
    else:
        random_text = text
    
    return random_text.strip()

# Specify the path to your PDF file
pdf_path = 'path/to/your/pdf/file.pdf'  # Replace with the actual PDF file path

# Extract random text from the PDF using easyocr and pytesseract
extracted_text = extract_random_text_from_pdf(pdf_path)

# Print the extracted text
print("Extracted Text:")
print(extracted_text)
