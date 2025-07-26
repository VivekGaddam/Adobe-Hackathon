import pdfplumber
from pathlib import Path

def extract_chunks(pdf_path):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if len(text.strip()) < 20:
                continue
            chunks.append({
                "document": Path(pdf_path).name,
                "page": i + 1,
                "text": text.strip()
            })
    return chunks