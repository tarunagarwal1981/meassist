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

@st.cache(allow_output_mutation=True, show_spinner=False)
def generate_embeddings(text_list):
    """Generates embeddings for a list of text documents."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_list, show_progress_bar=True)
    return embeddings

@st.cache(allow_output_mutation=True, show_spinner=False)
def create_faiss_index(embeddings):
    """Creates a FAISS index for a set of document embeddings."""
    if embeddings is None or not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("Invalid or empty embeddings array.")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_documents(query, index, text_list, top_k=5):
    """Searches the index for the documents most similar to the query."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [(text_list[idx], distances[0][i], idx) for i, idx in enumerate(indices[0])]

def create_augmented_prompt(query, retrieved_documents, filenames, top_k=3, max_tokens=16384):
    """Creates an augmented prompt by combining the query with top retrieved documents."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompt_tokens = tokenizer.encode(query)
    available_tokens = max_tokens - len(prompt_tokens) - 50  # Reserve some tokens for query and other text
    instruction = "Give a detailed answer unless asked for a brief or concise or short answer."
    context = instruction

    for doc, _, idx in sorted(retrieved_documents, key=lambda x: x[1])[:top_k]:
        doc_tokens = tokenizer.encode(doc)
        if len(doc_tokens) < available_tokens:
            context += " " + doc
            prompt_tokens += doc_tokens
            available_tokens = max_tokens - len(prompt_tokens)
        if available_tokens <= 0:
            break

    return f"Based on the following information: {context}\n\nAnswer the question: {query}", [filenames[i] for _, _, i in retrieved_documents[:top_k]]

def generate_response_with_gpt(augmented_prompt, sources, temperature=0.1, max_tokens=500, top_p=1.0):
    """Generates a response using the OpenAI ChatCompletion API with additional parameters."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": augmented_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    return response.choices[0].message['content'] + "\n\nSources: " + ", ".join(sources)

def analyze_reports(document_texts, filenames):
    """Analyzes the reports using the knowledge base files and provides recommendations."""
    analysis_prompt = "Analyze the provided reports based on the following logics:\n\n"
    analysis_prompt += "Data Comparison:\n"
    analysis_prompt += "- Obtain current operational data and compare against shop test baseline values\n"
    analysis_prompt += "- Interpolate shop test data for intermediate load points not directly measured\n"
    analysis_prompt += "- Calculate deviations in Pmax, Pcomp, exhaust temps, scavenge pressure, etc.\n"
    analysis_prompt += "- Utilize equipment failure mode database to relate deviations to potential faults\n\n"
    analysis_prompt += "Allowable Variances:\n"
    analysis_prompt += "- Pmax should be within ±3 bar across all units\n"
    analysis_prompt += "- Pcomp should be within ±3 bar across all units\n"
    analysis_prompt += "- Indicated pressure (Pi) should be within ±0.5 bar of unit average\n"
    analysis_prompt += "- Exhaust temps should be within ±50°C across all units\n\n"
    analysis_prompt += "Specific Failure Indicators:\n"
    analysis_prompt += "- Pmax lower than baseline suggests cylinder wear, valve issues, injection problems\n"
    analysis_prompt += "- Pcomp higher than baseline points to air path restrictions, fouling\n"
    analysis_prompt += "- Elevated exhaust temps can indicate mistimed injection or blow-by leakage\n"
    analysis_prompt += "- Fuel index >5% higher than baseline suggests worn pump plunger barrels\n"
    analysis_prompt += "- Specific lube oil consumption (SLOC) above 1.1 gm/kWhr indicates oil carry over\n\n"
    analysis_prompt += "Drain Oil Analysis:\n"
    analysis_prompt += "- Integrate wear metal findings, especially iron (Fe), with unit performance\n"
    analysis_prompt += "- Relate higher Fe levels to units showing Pmax, Pcomp, temp deviations\n"
    analysis_prompt += "- Confirms alignment between mechanical wear and combustion impacts\n\n"
    analysis_prompt += "Lube Oil Analysis:\n"
    analysis_prompt += "- Review lube oil analysis reports for any flagged anomalies or out-of-spec findings\n"
    analysis_prompt += "- Pay close attention to wear metals like iron (Fe), chromium (Cr), lead (Pb), copper (Cu)\n"
    analysis_prompt += "  - Elevated Fe can indicate liner, ring, or valve guide wear\n"
    analysis_prompt += "  - High Cr suggests piston ring or cylinder liner wear\n"
    analysis_prompt += "  - Increased Pb and Cu can point to bearing wear or coolant leakage\n"
    analysis_prompt += "- Evaluate oil viscosity, total base number (TBN), and oxidation for degradation\n"
    analysis_prompt += "  - Viscosity increase may indicate oil oxidation or soot loading\n"
    analysis_prompt += "  - TBN decrease suggests acid accumulation and reduced detergency\n"
    analysis_prompt += "  - Oxidation can lead to sludge formation and deposits\n"
    analysis_prompt += "- Correlate oil analysis findings with engine performance deviations\n"
    analysis_prompt += "  - Units with high wear metals and poor oil health may show reduced combustion metrics\n"
    analysis_prompt += "  - Relate oil degradation to potential for piston, ring deposits and sticking\n"
    analysis_prompt += "- Trend oil analysis reports over time to identify progressive wear or contamination\n"
    analysis_prompt += "  - Increasing wear metal levels can indicate accelerated component deterioration\n"
    analysis_prompt += "  - Contaminants like coolant, fuel, or water can signal seal leakage or condensation\n"
    analysis_prompt += "- Compare oil analysis results to manufacturer limits and industry benchmarks\n"
    analysis_prompt += "  - Determine if wear metals, viscosity, TBN are within acceptable ranges\n"
    analysis_prompt += "  - Identify if oil drain intervals need to be adjusted based on degradation rate\n\n"
    analysis_prompt += "Turbocharger Assessment:\n"
    analysis_prompt += "- Compare scavenge pressure to shop test baseline for deterioration\n"
    analysis_prompt += "- Evaluate turbocharger inlet vs outlet ΔT for exhaust flow restrictions\n"
    analysis_prompt += "- Inspect turbine side for deposits, nozzle ring damage if ΔT exceeds baseline\n\n"
    analysis_prompt += "Engine Tuning Review:\n"
    analysis_prompt += "- Relate fuel index and variable injection timing (VIT) to shop test\n"
    analysis_prompt += "- Index >5% higher can compensate for leakage, suboptimal combustion\n"
    analysis_prompt += "- Adjust VIT and fuel index to balance across units and improve performance\n\n"
    analysis_prompt += "Inspection Recommendations:\n"
    analysis_prompt += "- Compression and leak testing to quantify cylinder, ring, valve condition\n"
    analysis_prompt += "- Fuel system inspection for delivery volume, timing, injector spray pattern\n"
    analysis_prompt += "- Lube oil analysis to monitor wear metal trends, contamination, and oil health\n"
    analysis_prompt += "- Inspect cylinder liners, piston rings, and bearings for wear based on oil analysis\n"
    analysis_prompt += "- Turbocharger servicing to clean deposits, measure clearances\n"
    analysis_prompt += "- Piston, liner, and ring inspection for blow-by indications\n\n"
    analysis_prompt += "Reporting Structure:\n"
    analysis_prompt += "- List each key observation with specific data comparisons to baseline\n"
    analysis_prompt += "- Discuss 2-3 most likely failure modes based on symptoms and deviations\n"
    analysis_prompt += "- Provide clear logic for relating observations to potential faults\n"
    analysis_prompt += "- Include detailed recommended actions to address each issue\n"
    analysis_prompt += "- Integrate drain oil, lube oil, wear metal, and other evidence to support conclusions\n\n"
    analysis_prompt += "Analyze the reports and provide recommendations, checks, potential issues, and potential failure modes based on the knowledge base files."

    augmented_prompt, sources = create_augmented_prompt(analysis_prompt, list(zip(document_texts, [0] * len(document_texts), range(len(document_texts)))), filenames)
    response = generate_response_with_gpt(augmented_prompt, sources)
    return response

def main():
    st.title("Engine Expert")
    vessel_name = st.text_input("What is the name of your vessel?")
    if vessel_name:
        document_texts, filenames = process_files_in_folder(folder_path, vessel_name)
        if not document_texts:
            st.write("No relevant files found for the specified vessel.")
        else:
            document_embeddings = generate_embeddings(document_texts)
            faiss_index = create_faiss_index(np.array(document_embeddings))

            query = st.text_input("Enter your question or type 'exit' to quit:", "")
            if query:
                if query.lower() == 'exit':
                    st.write("Exiting the program. Goodbye!")
                    st.stop()

                with st.spinner('Processing your query...'):
                    retrieved_docs = search_documents(query, faiss_index, document_texts)
                    if not retrieved_docs:
                        st.write("Sorry, no relevant information could be found for your question.")
                    else:
                        augmented_prompt, sources = create_augmented_prompt(query, retrieved_docs, filenames)
                        response = generate_response_with_gpt(augmented_prompt, sources)
                        st.write("Answer:", response)

            if st.button("Analyze Reports"):
                with st.spinner('Analyzing reports...'):
                    analysis_result = analyze_reports(document_texts, filenames)
                    st.write("Analysis Result:")
                    st.write(analysis_result)

if __name__ == "__main__":
    main()
