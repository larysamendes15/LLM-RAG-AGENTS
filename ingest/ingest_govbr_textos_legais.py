import os
import io
import time
import hashlib
import requests
from typing import List, Dict, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain >=0.2 uses langchain_core.documents.Document
try:
    from langchain_core.documents import Document
except Exception:
    from langchain.docstore.document import Document  # fallback

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb

# ----------------------------
# Config
# ----------------------------
PERSIST_DIR = os.getenv("PERSIST_DIR", "data/chroma_reforma_textos_legais")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "thenlper/gte-small")
COLLECTION = os.getenv("COLLECTION", "reforma_textos_legais")
RESET_COLLECTION = os.getenv("RESET_COLLECTION", "false").lower() in {"1", "true", "yes", "y"}

SOURCES = [
    # PDFs (slides e documentos)
    "https://www.gov.br/fazenda/pt-br/acesso-a-informacao/acoes-e-programas/reforma-tributaria/regulamentacao-da-reforma-tributaria/lei-geral-do-ibs-da-cbs-e-do-imposto-seletivo/apresentacoes/2024-04-23_regulamentacao-da-reforma-tributaria_completa.pdf",
    "https://www.gov.br/fazenda/pt-br/acesso-a-informacao/acoes-e-programas/reforma-tributaria/arquivos/perguntas-e-respostas-reforma-tributaria_.pdf",
    "https://www.gov.br/fazenda/pt-br/acesso-a-informacao/acoes-e-programas/reforma-tributaria/apresentacoes/2023-11-14_cartilha_reforma-tributaria_atualizada-pos-senado.pdf",
    "https://www.gov.br/fazenda/pt-br/acesso-a-informacao/acoes-e-programas/reforma-tributaria/regulamentacao-da-reforma-tributaria/lei-geral-do-ibs-da-cbs-e-do-imposto-seletivo/apresentacoes/2024-05-20-_regulamentacao-da-reforma_3a-reuniao-tecnica-cd_cashback-e-cesta-basica.pdf",
    "https://cfc.org.br/wp-content/uploads/2024/07/reforma_tributaria.pdf",
    # HTML (leis / páginas)
    "https://www.planalto.gov.br/ccivil_03/constituicao/emendas/emc/emc132.htm",
    "https://espacolegislacao.totvs.com/reforma-tributaria/",
]

UA = os.getenv(
    "HTTP_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
)
TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60"))

# ----------------------------
# Utils
# ----------------------------
def _http_get(url: str) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    return r

def _is_pdf_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")

def _hash_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

# ----------------------------
# PDF extraction
# ----------------------------
def _extract_pdf_with_pdfplumber(data: bytes, url: str) -> List[Document]:
    """Try extracting text + tables with pdfplumber (best for tabelas)."""
    import pdfplumber  # optional dependency
    docs: List[Document] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        meta_title = ""
        try:
            meta_title = (pdf.metadata or {}).get("Title") or ""
        except Exception:
            meta_title = ""
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # Extract tables as TSV blocks
            table_blocks = []
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for tb in tables or []:
                lines = ["\t".join([c or "" for c in row]) for row in tb]
                if any(any(c for c in row) for row in tb):
                    table_blocks.append("\n".join(lines))
            combined = text
            if table_blocks:
                combined += "\n\n[TABELA]\n" + "\n\n[TABELA]\n".join(table_blocks)
            if combined.strip():
                docs.append(Document(
                    page_content=combined,
                    metadata={
                        "source": url,
                        "page": i,
                        "title": meta_title,
                        "type": "pdf"
                    }
                ))
    return docs

def _extract_pdf_with_pypdf(data: bytes, url: str) -> List[Document]:
    """Fallback with pypdf (sem tabelas estruturadas)."""
    from pypdf import PdfReader
    docs: List[Document] = []
    reader = PdfReader(io.BytesIO(data))
    meta_title = ""
    try:
        meta_title = (reader.metadata or {}).get("/Title") or ""
    except Exception:
        meta_title = ""
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": url, "page": i, "title": meta_title, "type": "pdf"}
            ))
    return docs

def load_pdf(url: str) -> List[Document]:
    r = _http_get(url)
    data = r.content
    # Try pdfplumber first to capture tables; fallback to pypdf
    try:
        import pdfplumber  # noqa: F401
        return _extract_pdf_with_pdfplumber(data, url)
    except Exception:
        return _extract_pdf_with_pypdf(data, url)

# ----------------------------
# HTML extraction
# ----------------------------
def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "form", "svg"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup
    texts = []
    for el in main.find_all(["h1","h2","h3","h4","h5","h6","p","li","td","th"]):
        t = el.get_text(separator=" ", strip=True)
        if t:
            texts.append(t)
    return "\n".join(texts)

def load_html(url: str) -> List[Document]:
    r = _http_get(url)
    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    text = _clean_html(html)
    if not text.strip():
        return []
    return [Document(
        page_content=text,
        metadata={"source": url, "title": title, "type": "html"}
    )]

# ----------------------------
# Chunk, embed, persist
# ----------------------------
def chunk_documents(raw_docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE","1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP","150")),
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    return splitter.split_documents(raw_docs)

def persist_to_chroma(chunks: List[Document]) -> None:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    if RESET_COLLECTION:
        try:
            client.delete_collection(COLLECTION)
            print(f"[RESET] Coleção '{COLLECTION}' apagada.")
        except Exception as e:
            print(f"[RESET] Ignorando erro ao apagar coleção: {e}")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION,
    )
    print(f"[DONE] {len(chunks)} chunks gravados em Chroma @ {PERSIST_DIR} (coleção: {COLLECTION})")

# ----------------------------
# Main
# ----------------------------
def main():
    print(f"[START] Ingestão de {len(SOURCES)} fontes...")
    all_docs: List[Document] = []
    for url in SOURCES:
        t0 = time.time()
        try:
            if _is_pdf_url(url):
                docs = load_pdf(url)
            else:
                docs = load_html(url)
            all_docs.extend(docs)
            dt = time.time() - t0
            print(f"  • {url} → {len(docs)} docs (em {dt:.1f}s)")
        except Exception as e:
            print(f"  ! Falha em {url}: {e}")

    print(f"[INFO] Total bruto: {len(all_docs)} docs (antes do split)")
    chunks = chunk_documents(all_docs)
    print(f"[INFO] Total de chunks: {len(chunks)}")
    persist_to_chroma(chunks)

if __name__ == "__main__":
    main()
