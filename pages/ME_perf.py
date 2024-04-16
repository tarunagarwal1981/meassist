import streamlit as st
import pandas as pd
from pdfminer.high_level import extract_text
import openai

def load_excel_data(excel_file):
    df = pd.read_excel(excel_file)
    return df

def analyze_data(df):
    thresholds = {
        'pmax_threshold': 3,   # ±3 bar
        'pcomp_threshold': 3,  # ±3 bar
        'exhaust_temp_threshold': 50,  # ±50°C
        'sloc_max': 1.1,  # gm/kwhr
        'sfoc_max': 190,  # gm/kwhr
        'diff_pmax_pcomp_max': 40  # bar
    }
    
    analysis_results = {
        'Mean Pmax': df['Pmax'].mean(),
        'Mean Pcomp': df['Pcomp'].mean(),
        'Mean Exhaust Temp': df['ExhaustTemp'].mean()
    }
    
    # Applying thresholds
    analysis_results['Pmax Deviations'] = (df['Pmax'] - analysis_results['Mean Pmax']).abs() > thresholds['pmax_threshold']
    analysis_results['Pcomp Deviations'] = (df['Pcomp'] - analysis_results['Mean Pcomp']).abs() > thresholds['pcomp_threshold']
    analysis_results['Exhaust Temp Deviations'] = (df['ExhaustTemp'] - analysis_results['Mean Exhaust Temp']).abs() > thresholds['exhaust_temp_threshold']
    analysis_results['SLOC Exceeds'] = df['SLOC'] > thresholds['sloc_max']
    analysis_results['SFOC Exceeds'] = df['SFOC'] > thresholds['sfoc_max']
    analysis_results['Diff Pmax Pcomp'] = (df['Pmax'] - df['Pcomp']).abs() > thresholds['diff_pmax_pcomp_max']
    
    return analysis_results

def extract_pdf_text(pdf_file):
    text = extract_text(pdf_file)
    return text

def compare_data_with_llm(excel_summary, pdf_text):
    prompt = f"Given the operational data: {excel_summary} and the shop test data: {pdf_text}, " \
             "interpolate shop tests for corresponding operational loads, compare parameters, " \
             "and identify deviations. Suggest potential failure modes."
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1000
    )
    return response.choices[0].text

def generate_final_report(deviations, analysis_results, knowledge_base_results):
    prompt = f"Create a detailed analysis report based on operational deviations: {deviations}, " \
             f"initial analysis results: {analysis_results}, and insights from the knowledge base: {knowledge_base_results}"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1500
    )
    return response.choices[0].text

def main():
    st.title('Main Engine Performance Analysis')

    excel_file = st.file_uploader("Upload the Excel file for ME performance", type=['xlsx', 'xls'])
    pdf_file = st.file_uploader("Upload the PDF file for shop test data", type='pdf')

    if excel_file and pdf_file:
        df = load_excel_data(excel_file)
        excel_summary = analyze_data(df)
        st.write("Initial Analysis Results:", excel_summary)
        
        pdf_text = extract_pdf_text(pdf_file)
        st.text_area("Extracted Text from PDF", value=pdf_text, height=200)
        
        deviations = compare_data_with_llm(excel_summary, pdf_text)
        st.text_area("LLM Analysis and Comparison Results", value=deviations, height=200)
        
        # Placeholder: Query your knowledge base
        knowledge_base_results = "Simulated response from knowledge base based on deviations."
        
        final_report = generate_final_report(deviations, excel_summary, knowledge_base_results)
        st.text_area("Final Analysis Report", value=final_report, height=300)

if __name__ == '__main__':
    main()
