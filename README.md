# Assistente RAG + Agentes — Reforma Tributária (gov.br)

PoC open-source de um assistente **RAG + Agentes (LangGraph)** que responde sobre a **Reforma Tributária** usando **apenas documentos oficiais do gov.br** (página: Textos Legais da Lei Geral do IBS, da CBS e do Imposto Seletivo).

## Objetivo
- Indexar documentos oficiais (gov.br) da Reforma Tributária.
- Responder **com citações obrigatórias** (URL e/ou página) e **self-check anti‑alucinação**.
- Orquestrar **agentes** via **LangGraph**: Supervisor → Retriever → Answerer → Self‑Check → Safety.
- UI com **Streamlit**. LLM **local** via **Ollama**. Embeddings **HuggingFace**. Vector store **Chroma/FAISS**.

## Stack
- Python · LangChain · LangGraph · FAISS/Chroma · Ollama (Llama‑3.1‑8B/Qwen2.5‑7B/Mistral‑7B) · HuggingFace embeddings (`gte-small`/`bge-small`) · Streamlit.

## Arquitetura (grafo)
Supervisor → Retriever → Answerer → Self‑Check → Safety → (resposta)
- **Supervisor**: roteia intents (por ora, sempre RAG).
- **Retriever**: busca no vector store local (Chroma/FAISS).
- **Answerer**: gera resposta com citações a partir dos chunks.
- **Self‑Check**: verifica se cada sentença tem evidência; caso não, responde “não encontrado com base nas fontes indexadas”.
- **Safety**: insere disclaimers (informativo; não é consultoria jurídica).

## Dados (apenas gov.br)
Use o script em `ingest/ingest_govbr_textos_legais.py`. Ele coleta **somente** os links listados em:
- `https://www.gov.br/fazenda/.../lei-geral-do-ibs-da-cbs-e-do-imposto-seletivo/textos-legais`

Isso inclui (exemplos): **LC 214/2025**, **mensagem de veto**, **redações finais do PLP 68/2024**, **EC 132/2023** etc.

> **Dica:** Deixe as referências externas (Planalto/Câmara/Senado) ativas pois são **oficiais** e partem do gov.br. Se quiser restringir estritamente a `gov.br`, ajuste `ALLOWED_HOSTS` no script.

## Setup rápido

### 1) Ambiente local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# rode o Ollama em paralelo e baixe um modelo:
# https://ollama.com
ollama run llama3.1:8b  # baixa o modelo
```

### 2) Ingestão (indexar gov.br)
```bash
python ingest/ingest_govbr_textos_legais.py
```

### 3) Rodar a demo (Streamlit)
```bash
streamlit run app/streamlit_app.py
```

### (Opcional) Docker
```bash
docker build -t rag-reforma .
docker run --add-host=host.docker.internal:host-gateway -p 8501:8501 rag-reforma
```

## Avaliação (README obrigatório)
- Perguntas de teste: `eval/questions.json` (20–30).
- Script de avaliação: `eval/eval_ragas.py` — mede *faithfulness* e *answer relevancy* (RAGAS), além de latência média. Gere relatório em `eval/report.md`.

## Limites éticos
- Conteúdo **informativo**; não substitui consultoria jurídica/tributária.
- Citar sempre a **fonte oficial** e evitar extrapolações.
- Se não houver evidências no corpus, responder que **não encontrou**.

## Estrutura do repositório
```
├─ app/
│  └─ streamlit_app.py
├─ src/
│  ├─ graph.py
│  ├─ agents/
│  │  ├─ supervisor.py
│  │  ├─ retriever_agent.py
│  │  ├─ answer_agent.py
│  │  ├─ self_check_agent.py
│  │  └─ safety_agent.py
├─ ingest/
│  └─ ingest_govbr_textos_legais.py
├─ eval/
│  ├─ questions.json
│  ├─ eval_ragas.py
│  └─ report.md
├─ tests/
│  └─ test_graph.py
├─ data/ (persistência do vector store)
├─ Dockerfile
├─ requirements.txt
├─ LICENSE
└─ CITATION.cff
```

## Próximos passos
- Adicionar **reranking** (ex.: cross-encoder pequeno) para melhorar precisão.
- Implementar **router de intents** mais inteligente no Supervisor.
- Melhorar **extractive citations** (mostrar a passagem exata + página).
