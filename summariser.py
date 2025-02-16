import textwrap
from extraction import extract_text_from_pdf, extract_text_from_docx
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load Summarization Model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarize_text(abstract, intro, conclusion, max_length=800):
    """Summarizes extracted text using a transformer model with a structured prompt."""
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)

    # Generate structured prompt
    prompt = (
        "Summarize the key sections of this research paper while preserving technical accuracy.\n"
        "Provide insights into core concepts, research challenges, methodologies, findings, and conclusions.\n\n"
        f"### Abstract:\n{abstract}\n\n"
        f"### Introduction:\n{intro}\n\n"
        f"### Conclusion:\n{conclusion}\n\n"
        "Generate a structured summary capturing the most important aspects of this paper."
    )

    # Tokenize & truncate text if too long
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    
    summary_ids = model.generate(**inputs, max_length=max_length, min_length=300, do_sample=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return "\n".join(textwrap.wrap(summary, width=80))

def summarize_research_paper(file_path):
    """Extracts text from a document and generates a structured summary."""
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format! Please provide a PDF or DOCX.")

    # Extract and clean sections
    abstract = text.split("\n")[0] if text else "No abstract found."
    intro = text.split("\n")[1] if len(text.split("\n")) > 1 else "No introduction found."
    conclusion = text.split("\n")[-1] if text else "No conclusion found."

    return summarize_text(abstract, intro, conclusion, max_length=800)

