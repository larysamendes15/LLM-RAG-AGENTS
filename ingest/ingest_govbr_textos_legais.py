import os, requests, tempfile, time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

INDEX_URL = "https://www.gov.br/fazenda/pt-br/acesso-a-informacao/acoes-e-programas/reforma-tributaria/regulamentacao-da-reforma-tributaria/lei-geral-do-ibs-da-cbs-e-do-imposto-seletivo/textos-legais"
PERSIST_DIR = os.getenv("PERSIST_DIR", "data/chroma_reforma_textos_legais")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "thenlper/gte-small")

def collect_pdf_links():
    r = requests.get(INDEX_URL, timeout=60); r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    urls = []
    for a in soup.select("a[href]"):
        href = a["href"].strip()
        if href.startswith("#"):
            continue
        u = urljoin(INDEX_URL, href)
        if u.lower().endswith(".pdf"):
            urls.append(u)
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out

def main():
    print("[INGEST] Coletando PDFs do gov.br …")
    pdfs = collect_pdf_links()
    if not pdfs:
        print("[WARN] Nenhum PDF encontrado."); return

    docs = []
    for url in pdfs:
        print("[PDF]", url)
        r = requests.get(url, timeout=180); r.raise_for_status()
        fd, tmp = tempfile.mkstemp(suffix=".pdf")
        with os.fdopen(fd, "wb") as f:
            f.write(r.content)
        for d in PyPDFLoader(tmp).load():
            d.metadata["source"] = url
            d.metadata["title"] = os.path.basename(url)
            docs.append(d)
        time.sleep(0.5)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    print(f"[INGEST] {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name="reforma_textos_legais",
    )

    print(f"[DONE] Chroma salvo em {PERSIST_DIR}")

if __name__ == "__main__":
    main()
