import json
from datetime import datetime
from typing import List, Dict


def generate_travel_plan(
    outline_data: Dict[str, List[Dict]],
    refined_page_texts: Dict[str, Dict[int, str]],
    input_metadata: Dict
) -> Dict:
    # 1. Define keywords for scoring section relevance
    scoring_keywords = {
        "cities": 1,
        "things to do": 2,
        "activities": 2,
        "beach": 2,
        "nightlife": 2,
        "cuisine": 3,
        "food": 3,
        "wine": 3,
        "cooking": 3,
        "packing tips": 4,
        "tips and tricks": 4,
        "culture": 5,
        "history": 5,
    }

    # 2. Flatten all outlines into list with document names
    ranked_sections = []
    for doc_name, outline in outline_data.items():
        for entry in outline:
            section_title = entry["text"].strip().lower()
            rank = 99
            for keyword, importance in scoring_keywords.items():
                if keyword in section_title:
                    rank = importance
                    break
            if rank != 99:
                ranked_sections.append({
                    "document": doc_name,
                    "section_title": entry["text"].strip(),
                    "importance_rank": rank,
                    "page_number": entry["page"]
                })

    # 3. Sort by importance_rank and pick top 5
    ranked_sections.sort(key=lambda x: x["importance_rank"])
    top_sections = ranked_sections[:5]

    # 4. Extract corresponding refined text for those pages
    subsection_analysis = []
    for section in top_sections:
        doc = section["document"]
        page = section["page_number"]
        if doc in refined_page_texts and page in refined_page_texts[doc]:
            text = refined_page_texts[doc][page].strip()
            subsection_analysis.append({
                "document": doc,
                "refined_text": text,
                "page_number": page
            })

    # 5. Construct final output JSON
    final_output = {
        "metadata": {
            "input_documents": list(outline_data.keys()),
            "persona": input_metadata["persona"]["role"],
            "job_to_be_done": input_metadata["job_to_be_done"]["task"],
            "processing_timestamp": datetime.utcnow().isoformat()
        },
        "extracted_sections": top_sections,
        "subsection_analysis": subsection_analysis
    }

    return final_output
