from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from src.agents.supervisor import supervisor_route
from src.agents.retriever_agent import retrieve
from src.agents.answer_agent import answer
from src.agents.self_check_agent import self_check
from src.agents.safety_agent import apply_safety

class GraphState(BaseModel):
    question: str
    route: str = Field(default="")
    chunks: List[dict] = Field(default_factory=list)
    result: Dict[str, Any] = Field(default_factory=dict)
    check: Dict[str, Any] = Field(default_factory=dict)

def node_supervisor(state: GraphState) -> GraphState:
    state.route = supervisor_route(state.question)
    return state

def node_retriever(state: GraphState) -> GraphState:
    state.chunks = retrieve(state.question, k=5)
    return state

def node_answer(state: GraphState) -> GraphState:
    state.result = answer(state.question, state.chunks)
    return state

def node_selfcheck(state: GraphState) -> GraphState:
    state.check = self_check(state.result)
    return state

def node_safety(state: GraphState) -> GraphState:
    state.result = apply_safety(state.result)
    return state

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("supervisor", node_supervisor)
    g.add_node("retriever", node_retriever)
    g.add_node("answer", node_answer)
    g.add_node("selfcheck", node_selfcheck)
    g.add_node("safety", node_safety)

    g.set_entry_point("supervisor")
    g.add_edge("supervisor", "retriever")
    g.add_edge("retriever", "answer")
    g.add_edge("answer", "selfcheck")

    # Se falhar self-check, encerra com mensagem negativa (poderia reconsultar retriever)
    def route_after_check(state: GraphState):
        return "safety" if state.check.get("ok") else END

    g.add_conditional_edges("selfcheck", route_after_check, {"safety": "safety", END: END})
    g.add_edge("safety", END)
    return g.compile()
