import textwrap
from extraction import extract_text_from_docx, extract_text_from_pdf, preprocess_text
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load Summarization Model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarize_text(legal_text, max_length=800):
    """Summarizes preprocessed legal text using a transformer model with an efficient prompt for legal documents."""
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)

    # Efficient prompt for legal documents
    prompt = (
        "Summarize the legal document concisely, focusing on:\n"
        "- Key provisions, terms, and clauses\n"
        "- Rights and responsibilities of the parties\n"
        "- Legal implications and penalties\n"
        "- Enforcement and obligations\n\n"
        f"### Legal Document Text:\n{legal_text}\n\n"
        "Provide a brief and neutral summary highlighting the essential legal points."
    )

    # Tokenize & truncate text if too long
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    
    summary_ids = model.generate(**inputs, max_length=800, min_length=300, do_sample=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return "\n".join(textwrap.wrap(summary, width=80))

def summarize_legal_document(file_path):
    """Extracts text, preprocesses it, and generates a structured summary for legal documents."""
    # Extract text
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format! Please provide a PDF or DOCX.")

    # Preprocess text
    preprocessed_text = preprocess_text(text)

    # Summarize preprocessed text
    return summarize_text(preprocessed_text, max_length=800)

