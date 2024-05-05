import streamlit as st
import os
import numpy as np
import pandas as pd
from pdfminer.high_level import extract_text
from docx import Document
import openpyxl
from sentence_transformers import SentenceTransformer
import faiss
import openai
from transformers import GPT2Tokenizer
import fitz  # PyMuPDF

# Define the folder path for document processing
folder_path = 'docs'

def get_api_key():
    """Retrieve the API key from Streamlit secrets or environment variables."""
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

openai.api_key = get_api_key()

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        text = extract_text(pdf_path)
        if not text.strip():  # Check if extracted text is empty
            doc = fitz.open(pdf_path)
            text = ''
            for page in doc:
                text += page.get_text()
            doc.close()
        return text
    except Exception as e:
        st.error(f"Error extracting text from {pdf_path}: {e}")
        return None

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    try:
        doc = Document(docx_path)
        return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        st.error(f"Error extracting text from {docx_path}: {e}")
        return None

def extract_text_from_excel(excel_path):
    """Extracts text from an Excel file."""
    try:
        df = pd.concat(pd.read_excel(excel_path, sheet_name=None), ignore_index=True)
        return df.to_csv(index=False, header=False)
    except Exception as e:
        st.error(f"Error extracting text from {excel_path}: {e}")
        return None

@st.cache(allow_output_mutation=True, show_spinner=False)
def process_files_in_folder(folder_path, vessel_name):
    """Processes files in a specified folder and extracts texts."""
    document_texts = []
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if vessel_name.lower() in file.lower():
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
                    filenames.append(file)  # Keep track of file names for source attribution
    return document_texts, filenames

def analyze_lube_oil_report(lube_oil_report):
    """Analyzes the lube oil report."""
    analysis_prompt = "Analyze the provided lube oil report based on the following logics:\n\n"
    analysis_prompt += "Lube Oil Analysis:\n"
    analysis_prompt += "- Review lube oil analysis reports for any flagged anomalies or out-of-spec findings\n"
    analysis_prompt += "- Pay close attention to wear metals like iron (Fe), chromium (Cr), lead (Pb), copper (Cu)\n"
    analysis_prompt += "- Evaluate oil viscosity, total base number (TBN), and oxidation for degradation\n"
    analysis_prompt += "- Correlate oil analysis findings with engine performance deviations\n"
    analysis_prompt += "- Trend oil analysis reports over time to identify progressive wear or contamination\n"
    analysis_prompt += "- Compare oil analysis results to manufacturer limits and industry benchmarks\n\n"
    analysis_prompt += f"Lube Oil Report:\n{lube_oil_report}\n\n"
    analysis_prompt += "Provide your analysis of the lube oil report based on the given logics."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": analysis_prompt}],
        temperature=0.1,
        max_tokens=500,
        top_p=1.0
    )
    return response.choices[0].message['content']

def analyze_scavenge_drain_report(scavenge_drain_report):
    """Analyzes the scavenge drain report."""
    analysis_prompt = "Analyze the provided scavenge drain report based on the following logics:\n\n"
    analysis_prompt += "Drain Oil Analysis:\n"
    analysis_prompt += "- Integrate wear metal findings, especially iron (Fe), with unit performance\n"
    analysis_prompt += "- Relate higher Fe levels to units showing Pmax, Pcomp, temp deviations\n"
    analysis_prompt += "- Confirm alignment between mechanical wear and combustion impacts\n\n"
    analysis_prompt += f"Scavenge Drain Report:\n{scavenge_drain_report}\n\n"
    analysis_prompt += "Provide your analysis of the scavenge drain report based on the given logics."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": analysis_prompt}],
        temperature=0.1,
        max_tokens=500,
        top_p=1.0
    )
    return response.choices[0].message['content']

def compile_analyses(lube_oil_analysis, scavenge_drain_analysis, knowledge_base):
    """Compiles the analyses and compares them against the knowledge base."""
    compilation_prompt = "Compile the lube oil analysis and scavenge drain analysis, and compare them against the knowledge base:\n\n"
    compilation_prompt += f"Lube Oil Analysis:\n{lube_oil_analysis}\n\n"
    compilation_prompt += f"Scavenge Drain Analysis:\n{scavenge_drain_analysis}\n\n"
    compilation_prompt += f"Knowledge Base:\n{knowledge_base}\n\n"
    compilation_prompt += "Provide full feedback, anomalies, and failure modes based on the compiled analyses and knowledge base."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": compilation_prompt}],
        temperature=0.1,
        max_tokens=1000,
        top_p=1.0
    )
    return response.choices[0].message['content']

def main():
    st.title("Engine Expert")
    vessel_name = st.text_input("What is the name of your vessel?")
    if vessel_name:
        document_texts, filenames = process_files_in_folder(folder_path, vessel_name)
        if not document_texts:
            st.write("No relevant files found for the specified vessel.")
        else:
            lube_oil_report = None
            scavenge_drain_report = None
            knowledge_base = ""

            for text, filename in zip(document_texts, filenames):
                if "lube_oil" in filename.lower():
                    lube_oil_report = text
                elif "scavenge_drain" in filename.lower():
                    scavenge_drain_report = text
                else:
                    knowledge_base += text + "\n\n"

            if lube_oil_report and scavenge_drain_report:
                if st.button("Analyze Reports"):
                    with st.spinner('Analyzing reports...'):
                        lube_oil_analysis = analyze_lube_oil_report(lube_oil_report)
                        scavenge_drain_analysis = analyze_scavenge_drain_report(scavenge_drain_report)
                        compiled_analysis = compile_analyses(lube_oil_analysis, scavenge_drain_analysis, knowledge_base)
                        st.write("Compiled Analysis:")
                        st.write(compiled_analysis)
            else:
                st.write("Lube oil report or scavenge drain report not found for the specified vessel.")

if __name__ == "__main__":
    main()
