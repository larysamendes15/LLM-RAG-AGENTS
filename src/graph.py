from typing import List, Dict, Any
from pydantic import BaseModel
from langgraph.graph import StateGraph, END

from src.agents.retriever_agent import retrieve
from src.agents.answer_agent import answer
from src.agents.self_check_agent import self_check
from src.agents.safety_agent import apply_safety

class GraphState(BaseModel):
    question: str
    chunks: List[dict] = []
    result: Dict[str, Any] = {}
    check: Dict[str, Any] = {}

def node_retriever(state: GraphState) -> GraphState:
    state.chunks = retrieve(state.question, k=5); return state

def node_answer(state: GraphState) -> GraphState:
    state.result = answer(state.question, state.chunks); return state

def node_selfcheck(state: GraphState) -> GraphState:
    state.check = self_check(state.result); return state

def node_safety(state: GraphState) -> GraphState:
    state.result = apply_safety(state.result); return state

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("retriever", node_retriever)
    g.add_node("answer", node_answer)
    g.add_node("selfcheck", node_selfcheck)
    g.add_node("safety", node_safety)
    g.set_entry_point("retriever")
    g.add_edge("retriever","answer")
    g.add_edge("answer","selfcheck")
    g.add_conditional_edges("selfcheck", lambda s: "safety" if s.check.get("ok") else END, {"safety":"safety", END:END})
    g.add_edge("safety", END)
    return g.compile()
