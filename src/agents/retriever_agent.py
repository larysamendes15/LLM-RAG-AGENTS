from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Dict, Any, Tuple

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = os.getenv("PERSIST_DIR", "data/chroma_reforma_textos_legais")
COLLECTION = os.getenv("COLLECTION", "reforma_textos_legais")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-m3")


@lru_cache(maxsize=1)
def _emb():
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

@lru_cache(maxsize=1)
def _client():
    return chromadb.PersistentClient(path=PERSIST_DIR)

@lru_cache(maxsize=1)
def _vs():
    return Chroma(
        client=_client(),
        collection_name=COLLECTION,
        embedding_function=_emb(),
    )

def _to_payload(docs) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in docs or []:
        md = getattr(d, "metadata", {}) or {}
        out.append(
            {
                "content": getattr(d, "page_content", "") or "",
                "source": md.get("source"),
                "title": md.get("title"),
                "page": md.get("page"),
            }
        )
    return out


def retrieve_top1(query: str) -> List[Dict[str, Any]]:
    """
    Sempre retorna o top-1:
    - tenta relevance_scores (maior=melhor)
    - depois score como distância (menor=melhor)
    - por fim, similarity_search sem score
    """
    vs = _vs()

    # 1) relevância (maior = melhor)
    try:
        pairs: List[Tuple[Any, float]] = vs.similarity_search_with_relevance_scores(query, k=3)
        if pairs:
            doc, rel = pairs[0]
            try:
                print(f"[retrieve] rel={float(rel):.3f}  q='{query[:80]}'")
            except Exception:
                pass
            return _to_payload(list(map(lambda doc: doc[0], pairs)))
    except Exception:
        pass

    # 2) distância (menor = melhor)
    try:
        pairs = vs.similarity_search_with_score(query, k=1)
        if pairs:
            doc, dist = pairs[0]
            try:
                print(f"[retrieve] dist={float(dist):.3f} q='{query[:80]}'")
            except Exception:
                pass
            return _to_payload([doc])
    except Exception:
        pass

    # 3) sem score
    try:
        docs = vs.similarity_search(query, k=1)
        print("[retrieve] fallback para similarity_search (sem score)")
        return _to_payload(docs)
    except Exception:
        return []


try:
    _ = _vs()
    count = _vs()._collection.count() 
    print(f"[Chroma] coleção='{COLLECTION}' em '{PERSIST_DIR}' → {count} documentos")
except Exception as e:
    print(f"[Chroma] não foi possível contar documentos: {e}")
