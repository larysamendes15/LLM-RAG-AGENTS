import os
import time
import streamlit as st
from dotenv import load_dotenv

from src.graph import build_graph, GraphState

load_dotenv()

st.set_page_config(page_title="Assistente Reforma Tributária (gov.br)", layout="wide")
st.title("Assistente Reforma Tributária — gov.br (RAG + Agentes)")

with st.sidebar:
    st.header("Config")
    st.write("LLM via Ollama (execute `ollama serve` e baixe um modelo, ex.: `ollama run llama3.1:8b`).")
    st.text_input("OLLAMA_HOST", os.environ.get("OLLAMA_HOST","http://localhost:11434"))
    st.text_input("OLLAMA_MODEL", os.environ.get("OLLAMA_MODEL","llama3.1:8b"))
    st.text_input("Embeddings", os.environ.get("EMBEDDINGS_MODEL","thenlper/gte-small"))
    st.text_input("VectorStore", os.environ.get("VECTORSTORE","chroma"))
    st.text_input("Persist Dir", os.environ.get("PERSIST_DIR","data/chroma_reforma_textos_legais"))
    st.caption("Para indexar, rode `python ingest/ingest_govbr_textos_legais.py`.")

question = st.text_input("Faça sua pergunta (ex.: O que é IBS? Há cashback para CadÚnico?)")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if st.button("Perguntar") and question.strip():
    start = time.time()
    state = GraphState(question=question.strip())
    result_state = st.session_state.graph.invoke(state)
    elapsed = (time.time() - start) * 1000

    if result_state.check and not result_state.check.get("ok"):
        st.warning(result_state.check.get("message"))
    if result_state.result:
        st.markdown(result_state.result.get("answer",""))

        with st.expander("Fontes utilizadas"):
            for i, ch in enumerate(result_state.result.get("used_chunks", []), start=1):
                title = ch.get("title") or "(sem título)"
                src = ch.get("source") or ""
                page = ch.get("page")
                page_str = f" (p. {page})" if page is not None else ""
                st.markdown(f"**[{i}] {title}{page_str}**  
{src}")

    st.caption(f"Latência: {elapsed:.0f} ms")
