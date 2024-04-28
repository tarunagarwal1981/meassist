import easyocr
from pdf2image import convert_from_path
import os

def extract_text_from_pdf(pdf_path):
    # Convert PDF to images
    pages = convert_from_path(pdf_path)
    
    # Initialize easyocr reader
    reader = easyocr.Reader(['en'])  # Specify the language(s) for OCR
    
    # Extract text from each page
    text = ''
    for page in pages:
        # Save the page as a temporary image file
        temp_image_path = 'temp_page.jpg'
        page.save(temp_image_path, 'JPEG')
        
        # Extract text from the image using easyocr
        result = reader.readtext(temp_image_path)
        page_text = ' '.join([res[1] for res in result])
        
        # Append the page text to the overall text
        text += page_text + '\n'
        
        # Remove the temporary image file
        os.remove(temp_image_path)
    
    return text.strip()

# Specify the path to your PDF file
pdf_path = 'path/to/your/pdf/file.pdf'  # Replace with the actual PDF file path

# Extract text from the PDF using easyocr
extracted_text = extract_text_from_pdf(pdf_path)

# Print the extracted text
print("Extracted Text:")
print(extracted_text)
