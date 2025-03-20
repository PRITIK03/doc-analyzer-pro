import streamlit as st
import pdfplumber
from pdf2image import convert_from_bytes
import easyocr
from PIL import Image
import io
import docx
import os
import time
import random

import pandas as pd
import difflib
import httpx
import tiktoken
import asyncio
from openai import OpenAI
from dotenv import load_dotenv
import spacy
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://doc-analyzer-pro.streamlit.app",
        "X-Title": "Doc Analyzer Pro"
    }
)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize EasyOCR (GPU enabled)
reader = easyocr.Reader(['en'], gpu=True)

# Streamlit UI Setup
st.set_page_config(page_title="Doc Analyzer Pro", layout="wide")
st.markdown("<h1 style='text-align: center;'>📄 Smart Document Analyzer</h1>", unsafe_allow_html=True)

# Helper Functions
def extract_text(file):
    """Extract text from uploaded file with optimized PDF processing."""
    content = file.read()

    if file.name.endswith('.pdf'):
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = "\n".join(page.extract_text_simple() for page in pdf.pages if page.extract_text_simple())

            if not text.strip():
                images = convert_from_bytes(content)
                text = "\n".join(" ".join(reader.readtext(img, detail=0)) for img in images)

        except Exception as e:
            st.error(f"PDF processing error: {str(e)}")
            return None

    elif file.name.endswith(('.docx', '.doc')):
        try:
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join(para.text for para in doc.paragraphs)
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

def truncate_text(text, max_tokens=2000):
    """Truncate text to fit within the token limit for OpenAI API (smaller chunks for speed)."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)

    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)

    return text

async def call_openai_api(prompt, max_retries=5):
    """Make asynchronous OpenAI API requests with retry logic, but suppress errors."""
    truncated_prompt = truncate_text(prompt)
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": truncated_prompt}],
        "temperature": 0.3,
        "max_tokens": 300
    }

    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(max_retries):
            try:
                response = await client.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
                response.raise_for_status()
                api_response = response.json()

                # ✅ Check if response contains expected keys
                if "choices" in api_response and api_response["choices"]:
                    return api_response["choices"][0]["message"]["content"]

            except httpx.HTTPStatusError:
                pass  # 🔹 Suppress HTTP error messages

            except httpx.RequestError:
                pass  # 🔹 Suppress network error messages

            except Exception:
                pass  # 🔹 Suppress all other errors

            # 🕐 Wait before retrying
            wait_time = random.uniform(2, 5)
            time.sleep(wait_time)

        return "⚠️ Summary generation encountered some issues, but this is the best possible result."

def run_async_function(async_func, *args):
    """Run an async function inside a Streamlit-friendly way"""
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            return asyncio.run(async_func(*args))  
    except RuntimeError:
        return asyncio.run(async_func(*args))  

async def async_generate_summary(text_chunks):
    """Async function to generate summaries"""
    return await asyncio.gather(*[call_openai_api(f"Summarize this concisely:\n{chunk}") for chunk in text_chunks])

@st.cache_data
def generate_summary(text):
    """Generate summary asynchronously (Fixed for Streamlit)"""
    text_chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
    return "\n".join(filter(None, run_async_function(async_generate_summary, text_chunks)))

@st.cache_data
def cached_answer(question, text):
    """Cache answers to avoid redundant OpenAI calls (Fixed for Streamlit)"""
    return run_async_function(call_openai_api, f"Document content:\n{text}\n\nQuestion: {question}\nAnswer:")

def extract_entities(text):
    """Extract named entities using spaCy batch processing."""
    docs = list(nlp.pipe([text[:100000]]))
    return [{"Text": ent.text, "Label": ent.label_} for ent in docs[0].ents]

def compare_documents(text1, text2):
    """Compare two documents and highlight differences with color coding."""
    matcher = difflib.ndiff(text1.split(), text2.split())

    styled_diff = []
    for word in matcher:
        if word.startswith("- "):  
            styled_diff.append(f"<span style='color: red; text-decoration: line-through;'>{word[2:]}</span>")
        elif word.startswith("+ "):  
            styled_diff.append(f"<span style='color: green; font-weight: bold;'>{word[2:]}</span>")
        elif word.startswith("? "):  
            styled_diff.append(f"<span style='color: blue; font-weight: bold;'>{word[2:]}</span>")
        else:  
            styled_diff.append(word[2:])

    return " ".join(styled_diff)

# File Upload Section
uploaded_file = st.file_uploader("📤 Upload document (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    with st.spinner("🔄 Extracting text..."):
        doc_text = extract_text(uploaded_file)

    if doc_text:
        st.session_state['document_text'] = doc_text
        st.success(f"✅ Successfully processed {uploaded_file.name}!")

        # ✅ Display Extracted Text
        st.subheader("📄 Extracted Text")
        with st.expander("🔍 View Extracted Text (Click to Expand)"):
            st.text_area("", doc_text, height=300)

        # ✅ Generate Summary
        with st.spinner("🧠 Generating summary..."):
            summary = generate_summary(doc_text)

        st.subheader("📌 Summary (Auto-Generated)")
        st.write(summary)

                # ✅ Named Entity Recognition
        st.subheader("🔍 Named Entity Recognition")
        with st.spinner("Extracting named entities..."):
            entities = extract_entities(doc_text)
        
        if entities:
            with st.expander("📊 View Extracted Entities"):
                df = pd.DataFrame(entities)
                st.dataframe(df)
        else:
            st.info("No named entities found in the document.")

        # ✅ Q&A Section
        st.subheader("💬 Ask a Question")
        question = st.text_input("❓ Type your question here")
        if question:
            with st.spinner("🤖 Finding the answer..."):
                answer = cached_answer(question, doc_text)
            st.subheader("📢 Answer")
            st.write(answer)


# ✅ Document Comparison Section
st.subheader("📄 Compare Documents")
compare_file = st.file_uploader("📤 Upload second document for comparison", type=["pdf", "docx", "txt"])

if compare_file:
    with st.spinner("🔄 Extracting text from second document..."):
        text2 = extract_text(compare_file)

    if text2:
        with st.spinner("🔍 Comparing documents..."):
            diff_result = compare_documents(doc_text, text2)

        st.subheader("📊 Differences Highlighted")
        st.markdown(diff_result, unsafe_allow_html=True)
