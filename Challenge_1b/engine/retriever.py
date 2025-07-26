def extract_ranked_sections(results):
    ranked_sections = []
    for i, meta in enumerate(results["metadatas"][0]):
        ranked_sections.append({
            "document": meta["document"],
            "section_title": results["documents"][0][i][:60] + "...",
            "importance_rank": i + 1,
            "page_number": meta["page"]
        })
    return ranked_sections