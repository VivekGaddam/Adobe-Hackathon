import os
import json
import fitz  # PyMuPDF
import pdfplumber
import spacy
import jsonschema
import pytesseract
import time
import PIL.Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load output schema
with open("./sample_dataset/schema/output_schema.json", "r") as f:
    output_schema = json.load(f)


def extract_title(page):
    spans = []
    blocks = page.get_text("dict")["blocks"]
    page_height = page.rect.height
    page_width = page.rect.width

    for b in blocks:
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text or len(text) < 4:
                    continue
                bbox = span["bbox"]
                top, left = bbox[1], bbox[0]
                spans.append({
                    "text": text,
                    "size": span["size"],
                    "top": top,
                    "left": left,
                    "width": bbox[2] - bbox[0]
                })

    if not spans:
        return None

    top_spans = [s for s in spans if s["top"] < 0.2 * page_height]

    def score(span):
        center_bonus = 1 - abs((span["left"] + span["width"] / 2) - page_width / 2) / (page_width / 2)
        return span["size"] * 1.5 + center_bonus * 2 - span["top"] / page_height

    if top_spans:
        best = max(top_spans, key=score)
        return best["text"]

    largest = max(spans, key=lambda s: s["size"])
    return largest["text"]


def parse_headings(page, page_number):
    blocks = page.get_text("dict")["blocks"]
    font_size_freq = {}
    all_spans = []

    for b in blocks:
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                fs = round(span["size"], 1)
                font_size_freq[fs] = font_size_freq.get(fs, 0) + 1
                all_spans.append(span)

    if not font_size_freq:
        return []

    body_size = max(font_size_freq, key=font_size_freq.get)

    headings = []

    for span in all_spans:
        text = span["text"].strip()
        size = span["size"]

        if not text or len(text) < 3:
            continue

        if size >= body_size + 1.5 and len(text.split()) <= 12:
            headings.append({
                "level": "H1" if size > body_size + 2 else "H2",
                "text": text,
                "page": page_number + 1
            })

    return headings


# def parse_tables(plumber_page):
#     tables_data = []
#     tables = plumber_page.extract_tables()
#     for table in tables:
#         cleaned = [[cell.strip() if cell else "" for cell in row] for row in table]
#         if any(any(cell for cell in row) for row in cleaned):
#             tables_data.append(cleaned)
#     return tables_data


def is_text_unreliable(text):
    words = text.strip().split()
    return len(words) < 5 or text.count(" ") < 2


def ocr_page(page):
    pix = page.get_pixmap(dpi=300)
    img = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img)




# ----------- Single PDF Processing Logic -------------
def process_single_pdf(pdf_path):
    input_dir = Path("./sample_dataset/pdfs")
    output_dir = Path("./sample_dataset/output")
    start_time = time.time()

    try:
        doc = fitz.open(pdf_path)
        output_data = {
            "title": extract_title(doc[0]) or 'untitled',
            "outline": []
        }

        full_text_pages = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(doc):
                text = page.get_text("text").strip()

                if is_text_unreliable(text):
                    text = ocr_page(page)

                full_text_pages.append(text)
                output_data["outline"].extend(parse_headings(page, page_num))

                # if page_num < len(pdf.pages):
                #     output_data["tables"].extend(parse_tables(pdf.pages[page_num]))

        jsonschema.validate(instance=output_data, schema=output_schema)

        output_file = output_dir / f"{pdf_path.stem}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Processed {pdf_path.name} in {time.time() - start_time:.2f}s")
        return f"Success: {pdf_path.name}"

    except Exception as e:
        return f"Error processing {pdf_path.name}: {e}"


# ----------- Batch PDF Processor -------------
def process_pdfs():
    input_dir = Path("./sample_dataset/pdfs")
    output_dir = Path("./sample_dataset/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_dir.glob("*.pdf"))

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_single_pdf, pdf_files))

    for result in results:
        print(result)


# ----------- Entry Point -------------
if __name__ == "__main__":
    print("Starting processing pdfs")
    process_pdfs()
    print("Completed processing pdfs")
