from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection(name="docs")

def embed_chunks(chunks):
    for i, chunk in enumerate(chunks):
        emb = model.encode(chunk["text"]).tolist()
        collection.add(
            documents=[chunk["text"]],
            embeddings=[emb],
            metadatas=[{
                "document": chunk["document"],
                "page": chunk["page"]
            }],
            ids=[f"{chunk['document']}_p{chunk['page']}_{i}"]
        )

def query_top_k(query_text, k=5):
    emb = model.encode(query_text).tolist()
    return collection.query(query_embeddings=[emb], n_results=k)