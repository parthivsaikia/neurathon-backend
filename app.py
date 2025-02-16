from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from extraction import extract_text_from_pdf, extract_text_from_docx, extract_sections, clean_text
from summariser import summarize_text
import shutil
import os

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def greet_json():
    return {"Hello": "World!"}

@app.post("/analyze/")
def analyze_research_paper(file: UploadFile = File(...)):
    try:
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(temp_file_path)
        elif file.filename.endswith(".docx"):
            text = extract_text_from_docx(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        sections = extract_sections(text)
        abstract = clean_text(sections.get("abstract", "No abstract found."))
        intro = clean_text(sections.get("introduction", "No introduction found."))
        conclusion = clean_text(sections.get("conclusion", "No conclusion found."))

        summary = summarize_text(abstract, intro, conclusion)
        os.remove(temp_file_path)

        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))