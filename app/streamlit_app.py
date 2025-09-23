from __future__ import annotations

import os
import sys
import time
import unicodedata
from typing import List, Dict, Any

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
try:
    from src.graph import get_graph
except Exception as e:
    st.set_page_config(page_title="Aris — erro de import", page_icon="⚠️")
    st.error(
        f"Falha ao importar `src.graph.get_graph`: {e}\n\n"
        f"Dica: confirme se está rodando de `{PROJECT_ROOT}` e se a pasta `src/` existe."
    )
    st.stop()

os.environ.setdefault("PERSIST_DIR", os.path.join(PROJECT_ROOT, "data", "chroma_reforma_textos_legais"))
os.environ.setdefault("COLLECTION", "reforma_textos_legais")

APP_NAME = "Aris — Assistente da Reforma Tributária"
FALLBACK_TEXT = "Não foi encontrado contexto para a pergunta solicitada."

os.environ.setdefault("OLLAMA_KEEP_ALIVE", "10m")
os.environ.setdefault("OLLAMA_NUM_PREDICT", "528")

def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def is_fallback(answer: str) -> bool:
    """Checagem robusta do fallback, independente de acentos/pontuação final."""
    norm = _normalize(answer).rstrip(".! ")
    target = _normalize(FALLBACK_TEXT).rstrip(".! ")
    return norm == target

def ref_link_from_chunk(ch: Dict[str, Any]) -> str | None:
    if not ch:
        return None
    src = ch.get("source")
    pg = ch.get("page")
    if not src:
        return None
    return f"{src}" + (f"#page={pg}" if pg is not None else "")

st.set_page_config(page_title=APP_NAME, page_icon="🏛️", layout="wide")
st.markdown(
    """
    <style>
      :root{
        --aris-bg: #f0f0f0;
        --aris-panel: #ffffff;
        --aris-card: #f9f9f9;
        --aris-primary: #32CD32;
        --aris-accent: #007bff;
        --aris-text: #333333;
        --aris-subtle: #6c757d;
        --aris-user-bg: #e0f2f7;
        --aris-assist-bg: #fff8e1;
        --aris-chip-bg: rgba(255,193,7,0.2);
        --aris-chip-border: rgba(255,193,7,0.4);
      }
      .stApp {
        background: radial-gradient(600px 400px at 80% 80%, rgba(255,215,0,0.15), transparent 60%),
                    radial-gradient(500px 300px at 20% 20%, rgba(0,123,255,0.15), transparent 60%),
                    var(--aris-bg);
        color: var(--aris-text);
      }
      .aris-header h1 {
        font-weight: 800 !important;
        letter-spacing: .3px;
        margin-bottom: .25rem;
        background: linear-gradient(90deg, var(--aris-accent), var(--aris-primary));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      }
      .chat-wrap { padding: 0; max-width: 800px; margin: 0 auto; }
      [data-testid="stChatMessage"]:has(.user) { display: flex; flex-direction: row-reverse; }
      [data-testid="stChatMessage"]:has(.user) .stChatMessageContent{ align-items: flex-end; }
      .bubble { display:inline-block; max-width:95%; border-radius:18px; padding:14px 18px; margin:4px 0 12px 0;
                line-height:1.6; box-shadow:0 2px 5px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.05);
                border:1px solid rgba(0,0,0,0.05); }
      .bubble.user { background: var(--aris-user-bg); border-bottom-right-radius:4px; color: var(--aris-text);}
      .bubble.assistant { background: var(--aris-assist-bg); border-bottom-left-radius:4px; color: var(--aris-text);}
      .ref-title { color: var(--aris-subtle); font-size:.85rem; margin:0 0 4px 4px; font-weight:500; }
      .ref-chip { display:inline-block; background:var(--aris-chip-bg); border:1px solid var(--aris-chip-border);
                  color:var(--aris-text); padding:6px 12px; margin:0 8px 4px 0; border-radius:16px; font-size:.85rem; text-decoration:none !important; }
      section[data-testid="stBottom"] { bottom: 20px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Opções")
    if st.button("🗑️ Limpar conversa", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")

st.markdown(f"""<div class="aris-header"><h1>{APP_NAME}</h1></div>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def _graph():
    return get_graph()

graph = _graph()

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = []

st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
for m in st.session_state.messages:
    if m["text"] == "...": 
        continue
    role = m.get("role", "assistant")
    with st.chat_message("user" if role == "user" else "assistant", avatar="🧑‍💻" if role == "user" else "🏛️"):
        cls = "user" if role == "user" else "assistant"
        st.markdown(f'<div class="bubble {cls}">{m.get("text","")}</div>', unsafe_allow_html=True)
        refs: List[str] = m.get("refs") or []
        if refs:
            st.markdown('<div class="ref-title">Referência</div>', unsafe_allow_html=True)
            for r in refs[:1]:
                st.markdown(f'<a class="ref-chip" href="{r}" target="_blank">Fonte</a>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

prompt = st.chat_input("Pergunte algo (ex.: O que é cashback?)")

if prompt:
    st.session_state.messages.append({"role": "user", "text": prompt, "refs": []})
    st.session_state.messages.append({"role": "assistant", "text": "...", "refs": []})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["text"] == "...":
    last_prompt = st.session_state.messages[-2]["text"]

    with st.chat_message("assistant", avatar="🏛️"):
        with st.spinner("Consultando a base e elaborando a resposta..."):
            t0 = time.time()
            result = graph.invoke({"question": last_prompt.strip()})
            t1 = time.time()
            elapsed = t1 - t0

    answer = (result or {}).get("answer", "").strip()
    chunks = (result or {}).get("chunks") or []

    refs: List[str] = []
    if not is_fallback(answer) and chunks:
        link = ref_link_from_chunk(chunks[0])
        if link:
            refs = [link]

    st.session_state.messages.pop()
    st.session_state.messages.append({"role": "assistant", "text": answer or "—", "refs": refs})

    with st.chat_message("assistant"):
        st.caption(f"⏱️ {elapsed:.2f}s")

    st.rerun()
