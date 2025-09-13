from typing import List, Dict, Any
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

def get_vectorstore():
    vectorstore = os.environ.get("VECTORSTORE", "chroma").lower()
    persist_dir = os.environ.get("PERSIST_DIR", "data/chroma_reforma_textos_legais")
    faiss_dir = os.environ.get("FAISS_DIR", "data/faiss_reforma_textos_legais")
    embeddings_model = os.environ.get("EMBEDDINGS_MODEL", "thenlper/gte-small")
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)

    if vectorstore == "faiss":
        return FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings, collection_name="reforma_textos_legais")

def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    results = []
    for d in docs:
        meta = d.metadata or {}
        results.append({
            "content": d.page_content,
            "source": meta.get("source"),
            "title": meta.get("title"),
            "page": meta.get("page"),
            "doc_id": meta.get("doc_id"),
            "chunk_id": meta.get("chunk_id"),
            "type": meta.get("type"),
        })
    return results
