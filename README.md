# 📚 PDF Chatbot — AI-Powered Document Q&A

A web application that lets users upload PDF documents and ask questions about their contents using OpenAI GPT-4o-mini with RAG (Retrieval-Augmented Generation).

---

## 🚀 Live Demo
> [Add your deployed URL here after deployment]

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER BROWSER                          │
│                  (HTML + CSS + Vanilla JS)                   │
│   Upload PDFs  ──►  Chat Interface  ──►  Source Display     │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP (REST API)
┌────────────────────────────▼────────────────────────────────┐
│                   FastAPI Backend (Python)                    │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │pdf_processor│  │  embeddings  │  │      llm.py      │   │
│  │             │  │              │  │                  │   │
│  │ PyMuPDF     │  │sentence-     │  │ OpenAI GPT-4o-mini    │   │
│  │ Text Extract│  │transformers  │  │ gpt-4o-mini │   │
│  │ Chunking    │  │ all-MiniLM   │  │                  │   │
│  └──────┬──────┘  └──────┬───────┘  └──────────────────┘   │
│         │                │                   ▲              │
│         ▼                ▼                   │              │
│  ┌──────────────────────────────┐            │              │
│  │        ChromaDB              │ ───────────┘              │
│  │   (Local Vector Database)    │  Retrieve top-5 chunks    │
│  │   cosine similarity search   │                           │
│  └──────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/pdf-chatbot.git
cd pdf-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```
Your `.env` file should look like:
```
OPENAI_API_KEY=your_actual_key_here
```

### 5. Run the Application
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Open in Browser
```
http://localhost:8000
```

---

## 📐 Design Decisions

### Chunking Strategy
- **Method:** Sliding window word-based chunking
- **Chunk size:** 500 words per chunk
- **Overlap:** 100 words between consecutive chunks
- **Why:** Overlapping chunks ensure that context spanning chunk boundaries is not lost. 500 words balances between enough context for good answers and not exceeding embedding model limits.

### Embedding Model Choice
- **Model:** `all-MiniLM-L6-v2` (sentence-transformers)
- **Why:**
  - Runs 100% locally — no API cost
  - Fast inference (~50ms per query)
  - 384-dimensional vectors — compact and efficient
  - Excellent semantic similarity performance for English text
  - Only 80MB download

### Vector Database
- **Tool:** ChromaDB (persistent local storage)
- **Distance metric:** Cosine similarity
- **Why:** Zero setup, no cloud account needed, persists to disk, perfect for this scale

### Prompt Design
```
System role → Document assistant that answers only from context
Context block → Top-5 retrieved chunks with source labels
History block → Last 6 messages for conversational memory
Question → Current user question
Constraints → Must cite sources, say "not found" if missing
```

### Retrieval Approach
- **Type:** Dense vector search (semantic similarity)
- **Top-K:** 5 most relevant chunks per query
- **Flow:** Question → embed → cosine search in ChromaDB → retrieve top 5 → pass to LLM

### LLM
- **Model:** OpenAI GPT-4o-mini 1.5 Flash
- **Why:** Cost-effective, fast, strong reasoning on document QA tasks

---

## 📁 Project Structure

```
pdf-chatbot/
├── backend/
│   ├── main.py            # FastAPI routes & session management
│   ├── pdf_processor.py   # PDF text extraction & chunking
│   ├── embeddings.py      # Embedding generation & ChromaDB operations
│   └── llm.py             # OpenAI prompt building & answer generation
├── frontend/
│   └── index.html         # Complete single-file frontend UI
├── chroma_db/             # Auto-created vector storage (gitignored)
├── requirements.txt
├── render.yaml            # Render.com deployment config
├── .env.example
├── .gitignore
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/session/new` | Create a new chat session |
| POST | `/upload/{session_id}` | Upload PDF files |
| POST | `/chat` | Ask a question |
| GET | `/session/{session_id}/history` | Get chat history |
| DELETE | `/session/{session_id}` | Delete session |
| GET | `/health` | Health check |

---

## 🌐 Deployment (Render.com — Free)

1. Push code to GitHub (public repo)
2. Go to [render.com](https://render.com) and sign up
3. Click **New → Web Service**
4. Connect your GitHub repo
5. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variable: `OPENAI_API_KEY = your_actual_key`
7. Click **Deploy** — get your live URL!

---

## ✅ Features Implemented

- [x] PDF upload (up to 50MB)
- [x] Text extraction with PyMuPDF
- [x] Overlapping chunk strategy
- [x] Local embeddings (sentence-transformers)
- [x] ChromaDB vector storage
- [x] Semantic retrieval (top-5 chunks)
- [x] OpenAI GPT-4o-mini LLM answering
- [x] Source attribution (filename + page number)
- [x] Relevant excerpt display with match score
- [x] Chat history / conversational memory
- [x] Multiple PDF support
- [x] Session management
- [x] Clean dark-mode UI
- [x] Drag & drop file upload

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, FastAPI |
| PDF Parsing | PyMuPDF (fitz) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| LLM | OpenAI GPT-4o-mini 1.5 Flash |
| Frontend | HTML, CSS, Vanilla JS |
| Deployment | Render.com |
