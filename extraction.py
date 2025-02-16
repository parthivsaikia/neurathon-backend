import re
import nltk
from langchain.document_loaders import PyPDFLoader
from docx import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

nltk.download('punkt')



# ===========================
# 1. Extract Text from PDF using PyPDFLoader
# ===========================
def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text = "\n".join([page.page_content for page in pages])
    return text


# ===========================
# 2. Extract Text from DOCX
# ===========================
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


# ===========================
# 3. Extract Key Sections (Abstract, Intro, Conclusion)
# ===========================
def extract_sections(text):
    sections = {"abstract": "", "introduction": "", "conclusion": ""}

    # Normalize text to lowercase for easier matching
    text_lower = text.lower()

    # Define section headers (may vary across papers)
    patterns = {
        "abstract": r"(?:abstract|summary)\s*(.*?)\s*(?:introduction|1\.)",
        "introduction": r"(?:introduction|1\.)\s*(.*?)\s*(?:conclusion|discussion|results)",
        "conclusion": r"(?:conclusion|summary)\s*(.*?)$"
    }

    # Extract sections using regex
    for section, pattern in patterns.items():
        match = re.search(pattern, text_lower, re.DOTALL)
        if match:
            sections[section] = match.group(1)

    return sections


# ===========================
# 4. Clean & Preprocess Extracted Text
# ===========================
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\[[0-9]+\]', '', text)  # Remove citations like [1], [2]
    text = re.sub(r'[^a-zA-Z0-9.,!?;:\-\'"\s]', '', text)  # Remove special characters
    return text.strip()
