# 📊 RAG + Agentes — Reforma Tributária  

Este projeto implementa um **pipeline RAG (Retrieval-Augmented Generation)** com agentes especializados para responder perguntas sobre a **Reforma Tributária Brasileira**.  

---

## ⚡ Stack principal

- **ChromaDB** (vector store local)  
- **LangChain / LangGraph** para orquestração  
- **Ollama** como servidor de LLM  
- **Streamlit** para interface interativa  
- **Ragas** para avaliação automática das respostas  

---

## 🚀 Instalação

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📥 Ingestão de documentos

O script abaixo faz o parsing dos textos legais e gera o índice vetorial no ChromaDB:

```bash
python ingest/ingest_govbr_textos_legais.py
```

O índice será salvo localmente em:

```bash
data/chroma_reforma_textos_legais/
```

---

## 🧠 Baixar um modelo no Ollama

Use um modelo leve para perguntas e respostas:

```bash
export OLLAMA_MODEL=mistral:7b
ollama pull $OLLAMA_MODEL
```

---

## ▶️ Executar o modelo no Ollama

Inicie o servidor do Ollama com:

```bash
ollama run mistral:7b
```

---

## 💻 Interface Web (Streamlit)

Rode a aplicação com:

```bash
python -m streamlit run app/streamlit_app.py
```

Acesse no navegador: **http://localhost:8501**

---

## 📊 Avaliação com Ragas

O pipeline inclui avaliação automática da qualidade das respostas usando métricas como:

- **Faithfulness** → fidelidade ao contexto  
- **Answer Relevancy** → relevância da resposta  
- **Context Precision/Recall** → precisão e cobertura dos trechos recuperados  
- **Latência média**

Execute:

```bash
python -m eval.eval_ragas
```

Os relatórios serão gerados em:

```
eval/report.md
eval/raw_results.csv
```

---

## 📂 Estrutura do projeto

```bash
.
├── app/                  # UI em Streamlit
├── eval/                 # Scripts e relatórios de avaliação (Ragas)
├── ingest/               # Scripts de ingestão de documentos
├── src/agents/           # Implementação dos agentes (retriever, self-check, safety...)
├── data/                 # Índice Chroma local
└── requirements.txt
```

---

## 🧩 Agentes do pipeline

- **Retriever** → Busca os trechos relevantes no ChromaDB.  
- **Answerer** → Gera a resposta usando o LLM, incluindo citações.  
- **Self-Check** → Valida a resposta (similaridade, concisão, referências).  
- **Safety** → Formata a resposta final e adiciona um disclaimer.