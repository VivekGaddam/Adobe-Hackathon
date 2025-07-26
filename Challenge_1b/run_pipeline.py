import json
import sys
from pathlib import Path
from engine.chunker import extract_chunks
from engine.embedder import embed_chunks, query_top_k
from engine.retriever import extract_ranked_sections
from engine.summarizer import summarize_chunks
from engine.formatter import format_output


def run_pipeline(collection_path):
    with open(Path(collection_path) / "challenge1b_input.json") as f:
        input_json = json.load(f)

    all_chunks = []
    for doc in input_json["documents"]:
        pdf_path = Path(collection_path) / "PDFs" / doc["filename"]
        all_chunks.extend(extract_chunks(pdf_path))

    embed_chunks(all_chunks)

    query = input_json["persona"]["role"] + ", " + input_json["job_to_be_done"]["task"]
    results = query_top_k(query, k=5)

    ranked = extract_ranked_sections(results)
    summaries = summarize_chunks(results)

    final_json = format_output(input_json, ranked, summaries)
    with open(Path(collection_path) / "challenge1b_output.json", "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    print("✅ Output written to", Path(collection_path) / "challenge1b_output.json")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "Collection 1"
    run_pipeline(path)