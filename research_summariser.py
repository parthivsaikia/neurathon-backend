import textwrap
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from trace import extract_text_from_pdf, extract_text_from_docx
# Load Summarization Model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def clean_text(text):
    """Clean up the text to remove unnecessary parts."""
    text = re.sub(r'[^a-zA-Z0-9\s\.\,]', '', text)  # Remove special characters
    text = re.sub(r'\n+', '\n', text)  # Remove excessive newlines
    text = text.strip()
    return text

def summarize_text(abstract, intro, conclusion, max_length=800):
    """Summarizes the extracted text using the transformer model."""
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)

    # Generate structured prompt
    prompt = (
        f"Summarize the following key sections of this research paper. Focus on core concepts, methodologies, findings, and conclusions.\n\n"
        f"### Abstract:\n{abstract}\n\n"
        f"### Introduction:\n{intro}\n\n"
        f"### Conclusion:\n{conclusion}\n\n"
        "Summarize the main insights and technical findings of the paper."
    )

    # Clean the prompt before summarization
    cleaned_prompt = clean_text(prompt)

    # Tokenize & truncate text if too long
    inputs = tokenizer(cleaned_prompt, return_tensors="pt", max_length=1024, truncation=True)

    # Generate summary
    summary_ids = model.generate(**inputs, max_length=max_length, min_length=300, do_sample=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Return wrapped summary text
    return "\n".join(textwrap.wrap(summary, width=80))



def extract_sections(text):
    """Extract abstract, intro, and conclusion from text."""
    abstract, intro, conclusion = "No abstract found.", "No introduction found.", "No conclusion found."
    
    # Regex to try and extract the sections based on common patterns
    abstract_match = re.search(r"(abstract|summary)\s*(.?)\s(introduction|1\.)", text, re.DOTALL | re.IGNORECASE)
    intro_match = re.search(r"(introduction|1\.)\s*(.?)\s(conclusion|discussion|results)", text, re.DOTALL | re.IGNORECASE)
    conclusion_match = re.search(r"(conclusion|summary)\s*(.*?)$", text, re.DOTALL | re.IGNORECASE)
    
    if abstract_match:
        abstract = clean_text(abstract_match.group(2))
    if intro_match:
        intro = clean_text(intro_match.group(2))
    if conclusion_match:
        conclusion = clean_text(conclusion_match.group(2))
    
    return abstract, intro, conclusion

def summarize_research_paper(file_path):
    """Extract text from a document and generate a structured summary."""
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format! Please provide a PDF or DOCX.")

    # Extract and clean sections
    abstract, intro, conclusion = extract_sections(text)
    return summarize_text(abstract, intro, conclusion, max_length=800)

