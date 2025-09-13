from src.graph import build_graph, GraphState

def test_graph_runs_minimal():
    g = build_graph()
    out = g.invoke(GraphState(question='O que é IBS?'))
    assert out is not None
