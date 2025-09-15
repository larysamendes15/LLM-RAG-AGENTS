import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb

# busca os k melhores chunks no Chroma
def _vs():
    persist_dir = os.getenv("PERSIST_DIR", "data/chroma_reforma_textos_legais")
    em = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDINGS_MODEL","thenlper/gte-small"))

    client = chromadb.PersistentClient(path=persist_dir)
    return Chroma(
        client=client,
        collection_name="reforma_textos_legais",
        embedding_function=em,
    )

def retrieve(query: str, k: int = 5):
    docs = _vs().similarity_search(query, k=k)
    return [{
        "content": d.page_content,
        "source": d.metadata.get("source"),
        "title": d.metadata.get("title"),
        "page": d.metadata.get("page")
    } for d in docs]
