from __future__ import annotations
import hashlib
from pathlib import Path
import requests
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import DATA_DIR, PDF_PATH, PDF_URL, get_settings

def download_pdf(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 0 and not force:
        return PDF_PATH
    response = requests.get(PDF_URL, timeout=60, headers={"User-Agent": "agentic-ai-rag-assignment/1.0"})
    response.raise_for_status()
    if "pdf" not in response.headers.get("content-type", "").lower() and not response.content.startswith(b"%PDF"):
        raise ValueError("The downloaded file does not appear to be a PDF.")
    PDF_PATH.write_bytes(response.content)
    return PDF_PATH

def extract_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": page_number, "text": text})
    if not pages:
        raise ValueError("No extractable text was found in the PDF.")
    return pages

def build_chunks(pages: list[dict]) -> list[dict]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings["chunk_size"], chunk_overlap=settings["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        for chunk_index, chunk_text in enumerate(splitter.split_text(page["text"])):
            normalized = " ".join(chunk_text.split())
            if not normalized:
                continue
            chunk_id = hashlib.sha256(f'{page["page"]}:{chunk_index}:{normalized}'.encode()).hexdigest()[:24]
            chunks.append({"id": f"chunk-{chunk_id}", "text": normalized, "page": page["page"], "chunk_index": chunk_index, "source": PDF_URL})
    return chunks

def prepare_chunks(force_download: bool = False) -> list[dict]:
    return build_chunks(extract_pages(download_pdf(force=force_download)))
