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

     # Create prompt
    prompt = (
        "Summarize the key sections of this research paper while preserving technical accuracy.\n"
        "### Abstract:\n" + abstract + "\n\n"
        "### Introduction:\n" + intro + "\n\n"
        "### Conclusion:\n" + conclusion
    )

        # Tokenize and summarize
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    summary_ids = model.generate(**inputs, max_length=512, min_length=100, do_sample=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return "\n".join(textwrap.wrap(summary, width=80))