from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END


SELF_CHECK_ENABLED = os.getenv("SELF_CHECK_ENABLED", "1") == "1"
K_TOP = int(os.getenv("RETRIEVER_K", "5"))
CTX_CLIP_CHARS = int(os.getenv("CTX_CLIP_CHARS", "1200"))

#retrieve → answer → (self_check) → safety → END

try:
    from src.agents.retriever_agent import retrieve_topk as _retrieve
    _HAS_TOPK = True
except Exception:
    from src.agents.retriever_agent import retrieve_top as _retrieve
    _HAS_TOPK = False

from src.agents.answer_agent import answer_best
from src.agents.self_check_agent import SelfCheckAgent
from src.agents.safety_agent import SafetyAgent

_self_checker = SelfCheckAgent() if SELF_CHECK_ENABLED else None
_safety = SafetyAgent() 

class GraphState(TypedDict, total=False):
    question: str
    chunks: List[Dict[str, Any]]
    answer: str               
    self_check: str       
    timings: Dict[str, float]
    citations: List[str]    

def expand_query(q: str) -> str:
    return (q or "").strip()

def _clip_chunk(c: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(c, dict):
        return {"content": str(c)[:CTX_CLIP_CHARS]}
    out = dict(c)
    out["content"] = str(c.get("content", ""))[:CTX_CLIP_CHARS]
    return out

def _citations_from_chunks(chunks: List[Dict[str, Any]]) -> List[str]:
    """Gera lista simples de fontes (URL + #page quando existir), sem duplicar."""
    seen, cites = set(), []
    for c in chunks or []:
        src = c.get("source")
        if not src or src in seen:
            continue
        seen.add(src)
        pg = c.get("page")
        cites.append(src + (f"#page={pg}" if pg is not None else ""))
    return cites

def node_retrieve(state: GraphState) -> GraphState:
    t0 = time.perf_counter()
    q = expand_query(state.get("question", ""))

    if _HAS_TOPK:
        chunks = _retrieve(q, k=max(1, K_TOP))
    else:
        chunks = _retrieve(q)

    t1 = time.perf_counter()
    return {"chunks": chunks, "timings": {"retrieve": t1 - t0}}

def node_answer(state: GraphState) -> GraphState:
    t0 = time.perf_counter()
    chunks = state.get("chunks", []) or []
    topk = [_clip_chunk(c) for c in chunks[:max(1, K_TOP)]]
    ans = answer_best(state.get("question", ""), topk)
    t1 = time.perf_counter()

    timings = dict(state.get("timings", {}))
    timings["answer"] = t1 - t0
    citations = _citations_from_chunks(chunks[:max(1, K_TOP)])
    return {"answer": ans, "timings": timings, "citations": citations}

def node_self_check(state: GraphState) -> GraphState:
    t0 = time.perf_counter()
    if not SELF_CHECK_ENABLED or _self_checker is None:
        timings = dict(state.get("timings", {}))
        timings["self_check"] = 0.0
        return {"self_check": "skipped", "timings": timings}

    chunks = state.get("chunks", []) or []
    topk = chunks[:max(1, K_TOP)]
    chk = _self_checker.check({
        "question": state.get("question", ""),
        "answer": state.get("answer", ""),
        "chunks": topk,
    })
    t1 = time.perf_counter()

    timings = dict(state.get("timings", {}))
    timings["self_check"] = t1 - t0
    return {"self_check": chk, "timings": timings}

def node_safety(state: GraphState) -> GraphState:
    """Aplica bloqueios, insere disclaimers e anexa fontes. Sobrescreve `answer`."""
    t0 = time.perf_counter()
    payload = {
        "query": state.get("question", ""),
        "final_answer": state.get("answer", "") or "",
        "citations": state.get("citations", []) or [],
        "agent_logs": [],
    }
    out = _safety(payload)
    safe_answer = out.get("final_answer", state.get("answer", ""))
    t1 = time.perf_counter()

    timings = dict(state.get("timings", {}))
    timings["safety"] = t1 - t0
    return {"answer": safe_answer, "timings": timings}

@lru_cache(maxsize=1)
def get_graph():
    g = StateGraph(GraphState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("answer", node_answer)
    if SELF_CHECK_ENABLED:
        g.add_node("self_check", node_self_check)
    g.add_node("safety", node_safety)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "answer")

    if SELF_CHECK_ENABLED:
        g.add_edge("answer", "self_check")
        g.add_edge("self_check", "safety")
    else:
        g.add_edge("answer", "safety")

    g.add_edge("safety", END)
    return g.compile()
