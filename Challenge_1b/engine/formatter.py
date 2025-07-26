from datetime import datetime

def format_output(input_json, sections, summaries):
    return {
        "metadata": {
            "input_documents": [doc["filename"] for doc in input_json["documents"]],
            "persona": input_json["persona"]["role"],
            "job_to_be_done": input_json["job_to_be_done"]["task"],
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": sections,
        "subsection_analysis": summaries
    }