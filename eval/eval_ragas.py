import os, sys, time, json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph import get_graph, GraphState

from langchain_ollama import ChatOllama

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "mistral:7b") 

def make_ollama_judge() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_JUDGE_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        num_ctx=8192,
    )


def make_embeddings():
    model_name = os.getenv("RAGAS_EMBEDDINGS_MODEL", "BAAI/bge-m3")
    return HuggingFaceEmbeddings(model_name=model_name)

def run_eval():
    with open("eval/questions.json", "r", encoding="utf-8") as f:
        qs = json.load(f)

    graph = get_graph()

    rows = []
    for i, item in enumerate(qs, start=1):
        q = item["question"]
        gt = item.get("ground_truth_support", "")
        t0 = time.time()

        state = GraphState(question=q)
        out = graph.invoke(state)  
        elapsed_ms = int((time.time() - t0) * 1000)

        answer = (out.get("answer") or "").strip()
        chunks = out.get("chunks", []) or out.get("used_chunks", []) or []
        context_texts = []
        for c in chunks:
            txt = c.get("content", "")
            if isinstance(txt, str) and txt.strip():
                context_texts.append(txt)
        if not context_texts:
            context_texts = [""] 

        rows.append({
            "question": q,
            "answer": answer,
            "contexts": context_texts,
            "ground_truth": gt,
            "latency_ms": elapsed_ms,
        })
        print(f"[{i:02d}/{len(qs)}] {elapsed_ms} ms | '{q[:70]}...'")

    df = pd.DataFrame(rows)
    ds = Dataset.from_pandas(df[["question", "answer", "contexts", "ground_truth"]])

    judge = make_ollama_judge()
    embedder = make_embeddings()
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy],
        llm=judge,
        embeddings=embedder,
        run_config=RunConfig(timeout=300, max_workers=8)
    )
    print(result)
    df_result = result.to_pandas()
    df_result.to_csv('eval_ragas_result.csv', index=False, encoding="utf-8")

    df = df.merge(df_result, how='inner', left_on='question', right_on='user_input')

    os.makedirs("eval", exist_ok=True)
    df.to_csv("eval/raw_results.csv", index=False, encoding="utf-8")

    faith = float(df['faithfulness'].mean())
    relev = float(df['answer_relevancy'].mean())

    lat_stats = df["latency_ms"].describe()
    p50 = int(df["latency_ms"].quantile(0.5))
    p95 = int(df["latency_ms"].quantile(0.95))

    with open("eval/report.md", "w", encoding="utf-8") as f:
        f.write("# Resultado de Avaliação (RAGAS)\n\n")
        f.write(f"- **Amostras**: {len(df)}\n")
        f.write(f"- **Faithfulness (média)**: {faith:.3f}\n")
        f.write(f"- **Answer Relevancy (média)**: {relev:.3f}\n\n")

        f.write("## Latência (ms)\n")
        f.write(f"- média: {int(lat_stats['mean'])} | min: {int(lat_stats['min'])} | max: {int(lat_stats['max'])}\n")
        f.write(f"- p50: {p50} | p95: {p95}\n\n")

        f.write("## Amostras (primeiras 5)\n")
        for _, row in df.iterrows():
            f.write("### Pergunta\n")
            f.write(row["question"] + "\n\n")
            f.write("**Resposta (modelo):**\n\n")
            ans = row["answer"] or ""
            f.write(ans + "\n\n")
            f.write("**Ground truth:**\n\n")
            f.write((row["ground_truth"] or "") + "\n\n")
            f.write("Answer relevance: ")
            f.write(str(row["answer_relevancy"]) + "\n\n")
            f.write("Faithfulness: ")
            f.write(str(row["faithfulness"]) + "\n\n")
            f.write("**Contextos:**\n")
            for c in row["contexts"]:
                f.write(f"- {c}\n")
            f.write("\n---\n\n")

    print("[DONE] Avaliação salva em eval/report.md e eval/raw_results.csv")

if __name__ == "__main__":
    run_eval()
