import textwrap
from extraction import extract_text_from_pdf, extract_text_from_docx, preprocess_text
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load Summarization Model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarize_text(contract_text, max_length=800):
    """Summarizes preprocessed text using a transformer model with an efficient prompt for contract policies."""
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)

    # Efficient prompt for contract policies
    prompt = (
        "Summarize this contract policy concisely, focusing on:\n"
        "- Key clauses and obligations\n"
        "- Rights and responsibilities of involved parties\n"
        "- Important terms and conditions\n"
        "- Legal implications and enforcement details\n\n"
        f"### Contract Text:\n{contract_text}\n\n"
        "Provide a clear and concise summary in plain language."
    )

    # Tokenize & truncate text if too long
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    
    summary_ids = model.generate(**inputs, max_length=200, min_length=100, do_sample=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return "\n".join(textwrap.wrap(summary, width=80))

def summarize_contract_policy(file_path):
    """Extracts text, preprocesses it, and generates a structured summary for contract policies."""
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