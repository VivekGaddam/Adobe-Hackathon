import spacy
nlp = spacy.load("en_core_web_sm")

def summarize_chunks(results):
    refined = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        doc_nlp = nlp(doc)
        sents = list(doc_nlp.sents)
        summary = " ".join(str(s) for s in sents[:3])  # Top 3 sentences
        refined.append({
            "document": meta["document"],
            "refined_text": summary,
            "page_number": meta["page"]
        })
    return refined