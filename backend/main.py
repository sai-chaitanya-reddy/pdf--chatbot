from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
import os
import sys

sys.path.append(os.path.dirname(__file__))

from pdf_processor import extract_text_from_pdf, chunk_pages
from embeddings import embed_and_store_chunks, retrieve_relevant_chunks, delete_session_collection
from llm import ask_gemini

app = FastAPI(title="PDF Chatbot API", version="1.0.0")

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
# Stores: { session_id: { "files": [...], "chat_history": [...] } }
sessions: Dict[str, Dict] = {}


# ─── Request/Response Models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict]
    session_id: str

class SessionResponse(BaseModel):
    session_id: str
    message: str


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html"))


@app.post("/session/new", response_model=SessionResponse)
def create_session():
    """Create a new chat session."""
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = {
        "files": [],
        "chat_history": []
    }
    return {"session_id": session_id, "message": "Session created successfully"}


@app.post("/upload/{session_id}")
async def upload_pdf(session_id: str, files: List[UploadFile] = File(...)):
    """
    Upload one or more PDFs.
    - Validates file type and size (max 50MB)
    - Extracts text, chunks it, embeds and stores in ChromaDB
    """
    if session_id not in sessions:
        sessions[session_id] = {"files": [], "chat_history": []}

    MAX_SIZE = 50 * 1024 * 1024  # 50 MB
    results = []

    for file in files:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF file.")

        file_bytes = await file.read()

        # Validate file size
        if len(file_bytes) > MAX_SIZE:
            raise HTTPException(status_code=400, detail=f"{file.filename} exceeds 50MB limit.")

        # Extract text from PDF
        pages = extract_text_from_pdf(file_bytes, file.filename)
        if not pages:
            raise HTTPException(status_code=400, detail=f"Could not extract text from {file.filename}. It may be a scanned PDF.")

        # Chunk the pages
        chunks = chunk_pages(pages)

        # Embed and store in ChromaDB
        total_chunks = embed_and_store_chunks(chunks, session_id)

        sessions[session_id]["files"].append(file.filename)

        results.append({
            "filename": file.filename,
            "pages_extracted": len(pages),
            "chunks_created": total_chunks
        })

    return {
        "message": f"Successfully processed {len(files)} file(s)",
        "session_id": session_id,
        "details": results
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Answer a question using RAG:
    1. Embed the question
    2. Retrieve top relevant chunks from ChromaDB
    3. Send chunks + history to Gemini
    4. Return answer + sources
    """
    session_id = request.session_id

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please create a new session.")

    if not sessions[session_id]["files"]:
        raise HTTPException(status_code=400, detail="No PDFs uploaded in this session. Please upload a PDF first.")

    # Retrieve relevant chunks
    chunks = retrieve_relevant_chunks(request.question, session_id, top_k=5)

    # Get answer from Gemini
    chat_history = sessions[session_id]["chat_history"]
    result = ask_gemini(request.question, chunks, chat_history)

    # Update chat history
    sessions[session_id]["chat_history"].append({"role": "user", "content": request.question})
    sessions[session_id]["chat_history"].append({"role": "assistant", "content": result["answer"]})

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "session_id": session_id
    }


@app.get("/session/{session_id}/history")
def get_history(session_id: str):
    """Get full chat history for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session_id": session_id,
        "chat_history": sessions[session_id]["chat_history"],
        "files": sessions[session_id]["files"]
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Clean up a session and its vector data."""
    delete_session_collection(session_id)
    if session_id in sessions:
        del sessions[session_id]
    return {"message": "Session deleted successfully"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "PDF Chatbot API is running"}


# Serve frontend static files
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
