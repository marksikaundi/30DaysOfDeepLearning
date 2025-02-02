from fastapi import FastAPI, File, UploadFile, Form
from typing import List
import shutil
from pathlib import Path

app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(
    file: UploadFile = File(...),
    description: str = Form(...)
):
    # Save uploaded file
    with Path(f"uploads/{file.filename}").open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "description": description,
        "content_type": file.content_type
    }

@app.post("/uploadfiles/")
async def create_upload_files(
    files: List[UploadFile] = File(...),
    note: str = Form(...)
):
    return {
        "filenames": [file.filename for file in files],
        "note": note
    }