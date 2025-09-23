# src/graph.py
from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

# =========================
# Config
# =========================
# Habilita self-check quando variável for "1" (ajuste conforme seu .env)
SELF_CHECK_ENABLED = os.getenv("SELF_CHECK_ENABLED", "1") == "1"
# Top-k a ser pedido ao retriever e repassado ao LLM/self-check
K_TOP = int(os.getenv("RETRIEVER_K", "3"))
# Clip em caracteres aplicado a CADA chunk antes de ir ao LLM
CTX_CLIP_CHARS = int(os.getenv("CTX_CLIP_CHARS", "1200"))

# =========================
# Agentes
# =========================
# Tentamos importar uma função top-k; se não existir, caímos no top-1
try:
    from src.agents.retriever_agent import retrieve_topk as _retrieve
    _HAS_TOPK = True
except Exception:
    from src.agents.retriever_agent import retrieve_top1 as _retrieve
    _HAS_TOPK = False

from src.agents.answer_agent import answer_best
from src.agents.self_check_agent import SelfCheckAgent

# Instância do self-checker só se habilitado
_self_checker = SelfCheckAgent() if SELF_CHECK_ENABLED else None


# =========================
# Estado do grafo
# =========================
class GraphState(TypedDict, total=False):
    question: str
    chunks: List[Dict[str, Any]]
    answer: str
    self_check: str
    timings: Dict[str, float]


# =========================
# Utilidades
# =========================
def expand_query(q: str) -> str:
    """Gancho para expansão/normalização da consulta."""
    return (q or "").strip()


def _clip_chunk(c: Dict[str, Any]) -> Dict[str, Any]:
    """Clipa o conteúdo do chunk para CTX_CLIP_CHARS, preservando metadados."""
    if not isinstance(c, dict):
        return {"content": str(c)[:CTX_CLIP_CHARS]}
    content = str(c.get("content", ""))[:CTX_CLIP_CHARS]
    out = dict(c)
    out["content"] = content
    return out


# =========================
# Nós
# =========================
def node_retrieve(state: GraphState) -> GraphState:
    t0 = time.perf_counter()
    q = expand_query(state.get("question", ""))

    if _HAS_TOPK:
        chunks = _retrieve(q, k=max(1, K_TOP))  # usa retrieve_topk
    else:
        # fallback: função top-1 disponível
        chunks = _retrieve(q)

    t1 = time.perf_counter()
    return {"chunks": chunks, "timings": {"retrieve": t1 - t0}}


def node_answer(state: GraphState) -> GraphState:
    t0 = time.perf_counter()

    chunks = state.get("chunks", []) or []
    # Seleciona até K_TOP e clippa cada um
    topk = [ _clip_chunk(c) for c in chunks[:max(1, K_TOP)] ]

    # answer_best deve aceitar lista de chunks (top-k)
    ans = answer_best(state.get("question", ""), topk)

    t1 = time.perf_counter()
    timings = dict(state.get("timings", {}))
    timings["answer"] = t1 - t0
    return {"answer": ans, "timings": timings}


def node_self_check(state: GraphState) -> GraphState:
    t0 = time.perf_counter()

    if not SELF_CHECK_ENABLED or _self_checker is None:
        timings = dict(state.get("timings", {}))
        timings["self_check"] = 0.0
        return {"self_check": "skipped", "timings": timings}

    chunks = state.get("chunks", []) or []
    topk = chunks[:max(1, K_TOP)]  # usa top-k para o verificador

    chk = _self_checker.check(
        {
            "question": state.get("question", ""),
            "answer": state.get("answer", ""),
            "chunks": topk,
        }
    )

    t1 = time.perf_counter()
    timings = dict(state.get("timings", {}))
    timings["self_check"] = t1 - t0
    return {"self_check": chk, "timings": timings}


@lru_cache(maxsize=1)
def get_graph():
    g = StateGraph(GraphState)

    g.add_node("retrieve", node_retrieve)
    g.add_node("answer", node_answer)
    if SELF_CHECK_ENABLED:
        g.add_node("self_check", node_self_check)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "answer")

    if SELF_CHECK_ENABLED:
        g.add_edge("answer", "self_check")
        g.add_edge("self_check", END)
    else:
        g.add_edge("answer", END)

    return g.compile()
