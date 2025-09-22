from __future__ import annotations

import re
from functools import lru_cache
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from src.agents.retriever_agent import retrieve_top1
from src.agents.answer_agent import answer_best
from src.agents.self_check_agent import fast_self_check


# ----------------------------
# State
# ----------------------------
class GraphState(TypedDict, total=False):
    question: str
    chunks: List[Dict[str, Any]]
    answer: str
    self_check: Dict[str, Any]


# ----------------------------
# Query normalizer/expander
# ----------------------------
_ws = re.compile(r"\s+")
def _normalize(q: str) -> str:
    return _ws.sub(" ", (q or "").strip()).lower()

def expand_query(q: str) -> str:
    """
    Expande abreviações e siglas comuns do domínio:
    - 'oq' -> 'o que é'
    - IBS/CBS/IS -> nomes completos (inclui variação sem acento)
    """
    qn = _normalize(q)

    # mapeia siglas para descrições
    expansions: List[str] = []
    if re.search(r"\bcbs\b", qn):
        expansions += [
            "Contribuição sobre Bens e Serviços",
            "Contribuicao sobre Bens e Servicos",
        ]
    if re.search(r"\bibs\b", qn):
        expansions += [
            "Imposto sobre Bens e Serviços",
            "Imposto sobre Bens e Servicos",
        ]
    if re.search(r"\bis\b", qn):
        expansions += ["Imposto Seletivo"]

    # contexto geral útil para ancoragem
    if any(k in qn for k in ["ibs", "cbs", "imposto", "contribuicao", "reforma"]):
        expansions += ["Reforma Tributária EC 132", "Lei Geral do IBS e da CBS"]

    if expansions:
        qn = f"{qn} | " + " | ".join(expansions)
    return qn


# ----------------------------
# Nodes (modo único: best)
# ----------------------------
def node_retrieve(state: GraphState) -> GraphState:
    raw_q = state["question"]
    q = expand_query(raw_q)
    chunks = retrieve_top1(q)  # SEM THRESHOLD
    return {"chunks": chunks}

def node_answer(state: GraphState) -> GraphState:
    q = state["question"]
    chunks = state.get("chunks", [])
    top = chunks[0] if chunks else {"content": "", "source": ""}
    ans = answer_best(q, chunks)
    return {"answer": ans}

def node_self_check(state: GraphState) -> GraphState:
    ans = state.get("answer", "")
    chunks = state.get("chunks", [])
    allowed = [c.get("source") for c in chunks]
    chk = fast_self_check(ans, allowed_sources=allowed)
    return {"self_check": chk}


# ----------------------------
# Graph factory
# ----------------------------
@lru_cache(maxsize=1)
def get_graph():
    g = StateGraph(GraphState)

    g.add_node("retrieve", node_retrieve)
    g.add_node("answer", node_answer)
    g.add_node("self_check", node_self_check)
    
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "self_check")
    g.add_edge("self_check", END)

    return g.compile()


# ----------------------------
# CLI helper
# ----------------------------
def answer_question(question: str) -> GraphState:
    graph = get_graph()
    return graph.invoke({"question": question})

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "O que é a CBS?"
    out = answer_question(q)
    print("RESPOSTA:\n", (out.get("answer") or "").strip())
    print("\nREFERÊNCIA:")
    if out.get("chunks"):
        src = out["chunks"][0].get("source")
        pg = out["chunks"][0].get("page")
        if src:
            print("-", src + (f"#page={pg}" if pg is not None else ""))
    print("\nSELF-CHECK:", out.get("self_check"))
