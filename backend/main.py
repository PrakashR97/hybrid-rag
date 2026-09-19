from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import shutil
import os
from pydantic import BaseModel
from typing import List, Dict
from rag_engine import process_document, ask_question_stream

app = FastAPI(title="Hybrid RAG API")

# Now accepts previous chat history
class QueryRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
        
    process_document(file_location)
    os.remove(file_location)
    return {"message": f"Successfully processed {file.filename}"}

@app.post("/query")
async def query_system(req: QueryRequest):
    # Streams the generator directly to the frontend
    return StreamingResponse(
        ask_question_stream(req.query, req.history), 
        media_type="text/event-stream"
    )