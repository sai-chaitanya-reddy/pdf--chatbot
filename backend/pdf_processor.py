import fitz  # PyMuPDF
import re
from typing import List, Dict


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[Dict]:
    """
    Extract text from PDF file bytes.
    Returns list of dicts with page number and text content.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text = clean_text(text)
        if text.strip():
            pages.append({
                "page_number": page_num + 1,
                "text": text,
                "filename": filename
            })

    doc.close()
    return pages


def clean_text(text: str) -> str:
    """Clean extracted text."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def chunk_pages(pages: List[Dict], chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """
    Split page texts into overlapping chunks for better retrieval.
    
    Strategy:
    - chunk_size=500 words keeps chunks meaningful but not too large
    - overlap=100 words ensures context is not lost at chunk boundaries
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        words = page["text"].split()
        total_words = len(words)

        if total_words == 0:
            continue

        start = 0
        while start < total_words:
            end = min(start + chunk_size, total_words)
            chunk_text = " ".join(words[start:end])

            chunks.append({
                "chunk_id": f"{page['filename']}_p{page['page_number']}_c{chunk_id}",
                "text": chunk_text,
                "page_number": page["page_number"],
                "filename": page["filename"],
                "start_word": start,
                "end_word": end
            })

            chunk_id += 1
            if end == total_words:
                break
            start += chunk_size - overlap  # move forward with overlap

    return chunks
