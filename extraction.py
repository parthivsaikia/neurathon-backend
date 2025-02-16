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


