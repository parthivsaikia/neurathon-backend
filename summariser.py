import textwrap
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def summarize_text(abstract, intro, conclusion, model_name="facebook/bart-large-cnn"):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Use CPU
    device = "cpu"
    model = model.to(device)

    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)