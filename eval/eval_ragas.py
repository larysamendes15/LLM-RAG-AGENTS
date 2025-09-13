import os, time, json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Este script roda uma avaliação simplificada:
# - Carrega perguntas de eval/questions.json
# - Para cada pergunta, chama a app (graph) e coleta resposta + fontes
# - Mede métricas RAGAS (faithfulness, answer_relevancy)
# IMPORTANTE: para rodar, é preciso que o vector store já exista (ingest feito).

from src.graph import build_graph, GraphState

def run_eval():
    with open("eval/questions.json", "r", encoding="utf-8") as f:
        qs = json.load(f)

    graph = build_graph()
    rows = []
    for item in qs:
        q = item["question"]
        t0 = time.time()
        state = GraphState(question=q)
        out = graph.invoke(state)
        elapsed = time.time() - t0

        answer = out.result.get("answer","")
        sources = "\n".join([c.get("source","") for c in out.result.get("used_chunks", [])])

        rows.append({
            "question": q,
            "answer": answer,
            "contexts": [sources],
            "ground_truth": "",  # opcional: preencha com um trecho considerado "verdadeiro"
            "latency_ms": int(elapsed*1000),
        })

    df = pd.DataFrame(rows)
    ds = Dataset.from_pandas(df[["question", "answer", "contexts", "ground_truth"]])

    result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
    print(result)

    df.to_csv("eval/raw_results.csv", index=False, encoding="utf-8")
    with open("eval/report.md", "w", encoding="utf-8") as f:
        f.write("# Resultado de Avaliação (RAGAS)\n\n")
        f.write(str(result))
        f.write("\n\nMétricas adicionais:\n")
        f.write(df["latency_ms"].describe().to_string())
    print("[DONE] Avaliação salva em eval/report.md e eval/raw_results.csv")


if __name__ == "__main__":
    run_eval()
