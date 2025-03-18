# app.py
import streamlit as st
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import io
import docx
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import spacy

# Load environment variables
load_dotenv()

# Initialize OpenAI client for OpenRouter with headers
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://doc-analyzer-pro.streamlit.app",
        "X-Title": "Doc Analyzer Pro"
    }
)

# Load spaCy model for entity recognition
nlp = spacy.load("en_core_web_sm")

# Helper Functions
def extract_text(file):
    """Extract text from uploaded file with OCR fallback for PDFs"""
    content = file.read()
    
    if file.name.endswith('.pdf'):
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = '\n'.join([page.extract_text() for page in pdf.pages])
            
            if not text.strip():
                images = convert_from_bytes(content)
                text = '\n'.join([pytesseract.image_to_string(img) for img in images])
                
        except Exception as e:
            st.error(f"PDF processing error: {str(e)}")
            return None
    
    elif file.name.endswith(('.docx', '.doc')):
        try:
            doc = docx.Document(io.BytesIO(content))
            text = '\n'.join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.error(f"DOCX processing error: {str(e)}")
            return None
    
    else:
        try:
            text = content.decode()
        except Exception as e:
            st.error(f"Text decoding error: {str(e)}")
            return None
    
    return text

def openai_call(prompt):
    """Make request to OpenRouter API"""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# ... [Keep the rest of the code identical to previous version] ...

def extract_entities(text):
    """Extract named entities using spaCy"""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "Text": ent.text,
            "Label": ent.label_,
            "Start": ent.start_char,
            "End": ent.end_char
        })
    return entities

def compare_documents(text1, text2):
    """Compare two documents and highlight differences"""
    diff = []
    lines1 = text1.split('\n')
    lines2 = text2.split('\n')
    
    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        if line1 != line2:
            diff.append({
                "Line": i+1,
                "Document 1": line1,
                "Document 2": line2
            })
    return diff

# Streamlit UI
st.set_page_config(page_title="Doc Analyzer Pro", layout="wide")
st.title("📄 Smart Document Analyzer (OpenRouter)")

# File Upload Section
uploaded_file = st.file_uploader("Upload document (PDF/DOCX/TXT)", 
                               type=["pdf", "docx", "txt"])

if uploaded_file:
    with st.spinner("Processing document..."):
        doc_text = extract_text(uploaded_file)
        
    if doc_text:
        st.session_state['document_text'] = doc_text
        st.success(f"Successfully processed {uploaded_file.name}!")
        
        with st.expander("View Extracted Text"):
            st.text(doc_text[:5000] + ("..." if len(doc_text) > 5000 else "")) 

# Analysis Features
if 'document_text' in st.session_state:
    text = st.session_state['document_text']
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Summary", "Q&A", "Entity Recognition", "Document Comparison"]
    )
    
    if analysis_type == "Summary":
        st.subheader("Document Summarization")
        length = st.radio("Summary Length", ["Concise", "Detailed", "Comprehensive"], index=1)
        
        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                summary = openai_call(f"Summarize this document with {length.lower()} detail:\n{text}")
                if summary:
                    st.subheader("Summary")
                    st.write(summary)
                else:
                    st.error("Summary generation failed")

    elif analysis_type == "Q&A":
        st.subheader("Document Q&A System")
        question = st.text_input("Ask a question about the document")
        
        if question and st.button("Get Answer"):
            with st.spinner("Analyzing document..."):
                answer = openai_call(f"Document content:\n{text}\n\nQuestion: {question}\nAnswer:")
                if answer:
                    st.subheader("Answer")
                    st.write(answer)
                else:
                    st.error("Failed to generate answer")

    elif analysis_type == "Entity Recognition":
        st.subheader("Named Entity Recognition")
        if st.button("Extract Entities"):
            with st.spinner("Identifying entities..."):
                entities = extract_entities(text)
                if entities:
                    st.subheader("Extracted Entities")
                    df = pd.DataFrame(entities)
                    st.dataframe(df)
                else:
                    st.info("No entities found in the document")

    elif analysis_type == "Document Comparison":
        st.subheader("Document Comparison")
        compare_file = st.file_uploader("Upload second document (PDF)", 
                                      type=["pdf"])
        
        if compare_file:
            with st.spinner("Processing second document..."):
                text2 = extract_text(compare_file)
            
            if text2 and st.button("Compare Documents"):
                with st.spinner("Identifying differences..."):
                    diff = compare_documents(text, text2)
                    st.subheader(f"Found {len(diff)} differences")
                    if diff:
                        df = pd.DataFrame(diff)
                        st.dataframe(df)
                    else:
                        st.info("No differences found between the documents")