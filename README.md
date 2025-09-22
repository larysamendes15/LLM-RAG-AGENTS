# RAG + Agentes (Reforma Tributária) — **Chroma somente**

Este pacote usa **Chroma** como vector store (sem FAISS/Pinecone).

## Como rodar
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 1) Ingestão (gera índice Chroma local)
python ingest/ingest_govbr_textos_legais.py

# 2) Baixe um modelo no Ollama (leve)
OLLAMA_MODEL=llama3.1:8b

#3) rodar o modelo
ollama run llama3.1:8b

# 3) UI
python -m streamlit run app/streamlit_app.py

# 4) Ragas
python -m eval.eval_ragas 
```

