# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from research_summariser import summarize_research_paper
from contract_summariser import summarize_contract_policy
from legal_summariser import summarize_legal_document
import shutil
import os

app = FastAPI()

origins = [
    "http://localhost:5173",
    "https://neurathon-frontend.vercel.app/",
    "https://bilahigpt.vercel.app/"
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
    return {"message": "Hello, World!","lol": "no"}

@app.post("/analyze/research/")
async def analyze_research_paper(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Generate summary using the summariser module
        summary = summarize_research_paper(temp_file_path)

        # Remove the temporary file
        os.remove(temp_file_path)

        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/analyze/contract/")
async def analyze_contract_paper(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Generate summary using the summariser module
        summary = summarize_contract_policy(temp_file_path)

        # Remove the temporary file
        os.remove(temp_file_path)

        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/analyze/legal/")
async def analyze_legal_paper(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Generate summary using the summariser module
        summary = summarize_legal_document(temp_file_path)

        # Remove the temporary file
        os.remove(temp_file_path)

        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
