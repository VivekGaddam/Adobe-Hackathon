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
def parse_headings(page, page_number, body_size, recurring_texts):
    blocks = page.get_text("dict")["blocks"]
    all_spans = []

    for b in blocks:
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                all_spans.append(span)

    headings = []

    for span in all_spans:
        text = span["text"].strip()
        size = span["size"]

        if not text or len(text) < 3:
            continue

        # --- KEY CHANGE: Added 'and text not in recurring_texts' ---
        is_heading = (
            size >= body_size + 1.5 and
            len(text.split()) <= 12 and
            text not in recurring_texts # This filters out headers and footers
        )

        if is_heading:
            headings.append({
                "level": "H1" if size > body_size + 2 else "H2",
                "text": text,
                "page": page_number + 1
            })

    return headings

# Unchanged helper functions
def is_text_unreliable(text):
    words = text.strip().split()
    return len(words) < 5 or text.count(" ") < 2

def ocr_page(page):
    pix = page.get_pixmap(dpi=300)
    img = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img)

# --- MODIFIED to perform frequency analysis first ---
def process_single_pdf(pdf_path):
    input_dir = Path("./sample_dataset/pdfs")
    output_dir = Path("./sample_dataset/output")
    start_time = time.time()

    try:
        doc = fitz.open(pdf_path)

        # <<< START: NEW FREQUENCY COUNTING LOGIC >>>
        # Pass 1: Determine body size and find all recurring text (headers/footers)
        font_size_counts = Counter()
        line_text_counts = Counter()
        
        # Get body size based on character count per font size
        for page in doc:
            for block in page.get_text("dict")['blocks']:
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        size = round(span['size'], 1)
                        text = span['text'].strip()
                        font_size_counts.update({size: len(text)})
        
        body_size = font_size_counts.most_common(1)[0][0] if font_size_counts else 12

        # Count text frequency to identify headers/footers
        for page in doc:
            for block in page.get_text("dict")['blocks']:
                for line in block.get('lines', []):
                    line_text = "".join(s['text'] for s in line.get('spans', [])).strip()
                    if line_text:
                        line_text_counts.update([line_text])

        # Any text that appears on more than 2 pages is likely a header or footer
        recurring_texts = {text for text, count in line_text_counts.items() if count > 2}
        # <<< END: NEW FREQUENCY COUNTING LOGIC >>>

        # Pass 2: Process the document for real headings
        output_data = {
            "title": extract_title(doc[0]) or 'untitled',
            "outline": []
        }

        # This outer with-block for pdfplumber seems unused in your core logic,
        # but is kept to maintain your original structure.
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(doc):
                text = page.get_text("text").strip()

                if is_text_unreliable(text):
                    text = ocr_page(page)

                # Now, call parse_headings with the new information
                output_data["outline"].extend(parse_headings(page, page_num, body_size, recurring_texts))

        jsonschema.validate(instance=output_data, schema=output_schema)

        output_file = output_dir / f"{pdf_path.stem}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Processed {pdf_path.name} in {time.time() - start_time:.2f}s")
        return f"Success: {pdf_path.name}"

    except Exception as e:
        return f"Error processing {pdf_path.name}: {e}"

# --- This section is unchanged ---
def process_pdfs():
    input_dir = Path("./sample_dataset/pdfs")
    output_dir = Path("./sample_dataset/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_dir.glob("*.pdf"))

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_single_pdf, pdf_files))

    for result in results:
        print(result)

if __name__ == "__main__":
    print("Starting processing pdfs")
    process_pdfs()
    print("Completed processing pdfs")