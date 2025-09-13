import os
import re
import time
import tempfile
import hashlib
import unicodedata
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

TEXTOS_LEGAIS_URL = os.environ.get(
    "TEXTOS_LEGAIS_URL",
    "https://www.gov.br/fazenda/pt-br/acesso-a-informacao/acoes-e-programas/reforma-tributaria/regulamentacao-da-reforma-tributaria/lei-geral-do-ibs-da-cbs-e-do-imposto-seletivo/textos-legais",
)

ALLOWED_HOSTS = {
    "www.gov.br", "gov.br",
    "www.planalto.gov.br", "planalto.gov.br",
    "www.camara.leg.br", "camara.leg.br",
    "www12.senado.leg.br", "www25.senado.leg.br", "senado.leg.br"
}

VECTORSTORE = os.environ.get("VECTORSTORE", "chroma").lower()
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "thenlper/gte-small")
PERSIST_DIR = os.environ.get("PERSIST_DIR", "data/chroma_reforma_textos_legais")
FAISS_DIR = os.environ.get("FAISS_DIR", "data/faiss_reforma_textos_legais")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Ingest/1.0)"}

def normalize_space(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", s or "").strip()

def sha12(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]

def collect_links(index_url: str):
    r = requests.get(index_url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.select("a[href]"):
        href = a["href"].strip()
        if href.startswith("#"):
            continue
        url = urljoin(index_url, href)
        host = urlparse(url).netloc.lower()
        if host not in ALLOWED_HOSTS:
            continue
        title = normalize_space(a.get_text(" ").strip()) or url
        links.append({"url": url, "title": title})

    dedup = []
    seen = set()
    for it in links:
        if it["url"] in seen: 
            continue
        seen.add(it["url"])
        dedup.append(it)
    return dedup

def is_pdf(url: str) -> bool:
    return url.lower().endswith(".pdf")

def load_pdf(url: str, title: str):
    print(f"[PDF] {title} <- {url}")
    with requests.get(url, headers=HEADERS, timeout=120, stream=True) as r:
        r.raise_for_status()
        fd, tmp = tempfile.mkstemp(suffix=".pdf")
        with os.fdopen(fd, "wb") as f:
            for chunk in r.iter_content(1024*512):
                if chunk: f.write(chunk)
    loader = PyPDFLoader(tmp)
    docs = loader.load()
    for d in docs:
        d.metadata.update({"source": url, "title": title, "type": "pdf"})
    return docs

def load_html(url: str, title: str):
    print(f"[HTML] {title} <- {url}")
    loader = WebBaseLoader(url)
    docs = loader.load()
    for d in docs:
        d.metadata.update({"source": url, "title": title, "type": "html"})
    return docs

def main():
    print("[STEP] Coletando links do gov.br (Textos Legais)...")
    items = collect_links(TEXTOS_LEGAIS_URL)

    KEYWORDS = [
        "lei complementar", "plp", "projeto de lei complementar",
        "mensagem de veto", "emenda constitucional", "redação final",
        "parecer", "substitutivo"
    ]
    filtered = []
    for it in items:
        t = (it["title"] or "").lower()
        if any(k in t for k in KEYWORDS) or is_pdf(it["url"]):
            filtered.append(it)
    if not filtered:
        filtered = items

    docs_all = []
    for it in filtered:
        try:
            docs = load_pdf(it["url"], it["title"]) if is_pdf(it["url"]) else load_html(it["url"], it["title"])
            docs = [d for d in docs if d.page_content and len(d.page_content.strip()) > 200]
            docs_all.extend(docs)
            time.sleep(0.6)
        except Exception as e:
            print("[WARN]", it["url"], e)

    if not docs_all:
        print("[ERROR] Nada carregado.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs_all)

    for i, d in enumerate(chunks):
        d.metadata.setdefault("doc_id", sha12(d.metadata.get("source","") + d.metadata.get("title","")))
        d.metadata.setdefault("chunk_id", i)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    os.makedirs(os.path.dirname(PERSIST_DIR), exist_ok=True)
    os.makedirs(os.path.dirname(FAISS_DIR), exist_ok=True)

    if VECTORSTORE == "faiss":
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(FAISS_DIR)
        print(f"[DONE] FAISS salvo em {FAISS_DIR}")
    else:
        vs = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR, collection_name="reforma_textos_legais")
        vs.persist()
        print(f"[DONE] Chroma salvo em {PERSIST_DIR}")

if __name__ == "__main__":
    main()
