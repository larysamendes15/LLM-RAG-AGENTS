import os, sys, time, json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_huggingface import HuggingFaceEmbeddings

# Permite rodar com: python -m eval.eval_ragas (a partir da raiz do projeto)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph import get_graph, GraphState

# =========================
# Config do LLM avaliador (Ollama)
# =========================
from langchain_ollama import ChatOllama

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "llama3.1:8b") 

def make_ollama_judge() -> ChatOllama:
    # Para avaliação, usar temperatura 0 (determinístico) e contexto amplo
    return ChatOllama(
        model=OLLAMA_JUDGE_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        num_ctx=8192,
    )


def make_embeddings():
# use o mesmo modelo de embeddings que você já usa no retriever,
# ou um default leve/estável:
    model_name = os.getenv("RAGAS_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # se quiser exatamente o mesmo do seu retriever, use:
    # model_name = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)

def run_eval():
    # 1) Carregar perguntas
    with open("eval/questions.json", "r", encoding="utf-8") as f:
        qs = json.load(f)

    # 2) Grafo da sua aplicação (retriever + answer)
    graph = get_graph()

    rows = []
    for i, item in enumerate(qs, start=1):
        q = item["question"]
        gt = item.get("ground_truth_support", "")
        t0 = time.time()

        # 3) Invocar grafo
        state = GraphState(question=q)
        out = graph.invoke(state)   # dict
        elapsed_ms = int((time.time() - t0) * 1000)

        # 4) Extrair resposta e contextos
        answer = (out.get("answer") or "").strip()
        chunks = out.get("chunks", []) or out.get("used_chunks", []) or []
        context_texts = []
        for c in chunks[:3]:                 # até top-3
            txt = c.get("content", "")
            if isinstance(txt, str) and txt.strip():
                context_texts.append(txt[:2000])  # limita para não estourar tokens
        if not context_texts:
            context_texts = [""]  # RAGAS requer lista não vazia

        rows.append({
            "question": q,
            "answer": answer,
            "contexts": context_texts,
            "ground_truth": gt,
            "latency_ms": elapsed_ms,
        })
        print(f"[{i:02d}/{len(qs)}] {elapsed_ms} ms | '{q[:70]}...'")

    # 5) DataFrame -> Dataset (formato esperado pelo RAGAS)
    df = pd.DataFrame(rows)
    ds = Dataset.from_pandas(df[["question", "answer", "contexts", "ground_truth"]])

    # 6) LLM avaliador (Ollama) e avaliação
    judge = make_ollama_judge()
    embedder = make_embeddings()
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy],
        llm=judge,
        embeddings=embedder,   # 👈 evita fallback para OpenAIEmbeddings
    )
    print(result)

    # 7) Persistir resultados
    os.makedirs("eval", exist_ok=True)
    df.to_csv("eval/raw_results.csv", index=False, encoding="utf-8")

    # Extrair médias (compat. com variações do ragas)
    try:
        faith = float(result["faithfulness"])
        relev = float(result["answer_relevancy"])
    except Exception:
        # fallback bem simples caso a estrutura varie
        faith = None
        relev = None

    lat_stats = df["latency_ms"].describe()
    p50 = int(df["latency_ms"].quantile(0.5))
    p95 = int(df["latency_ms"].quantile(0.95))

    with open("eval/report.md", "w", encoding="utf-8") as f:
        f.write("# Resultado de Avaliação (RAGAS)\n\n")
        f.write(f"- **Amostras**: {len(df)}\n")
        f.write(f"- **Faithfulness (média)**: {faith:.3f}\n" if isinstance(faith, (int, float,)) else "- **Faithfulness (média)**: n/a\n")
        f.write(f"- **Answer Relevancy (média)**: {relev:.3f}\n\n" if isinstance(relev, (int, float,)) else "- **Answer Relevancy (média)**: n/a\n\n")

        f.write("## Latência (ms)\n")
        f.write(f"- média: {int(lat_stats['mean'])} | min: {int(lat_stats['min'])} | max: {int(lat_stats['max'])}\n")
        f.write(f"- p50: {p50} | p95: {p95}\n\n")

        f.write("## Amostras (primeiras 5)\n")
        for _, row in df.head(5).iterrows():
            f.write("### Pergunta\n")
            f.write(row["question"] + "\n\n")
            f.write("**Resposta (modelo):**\n\n")
            ans = row["answer"] or ""
            f.write((ans[:1500] + ("...\n\n" if len(ans) > 1500 else "\n\n")))
            f.write("**Ground truth:**\n\n")
            f.write((row["ground_truth"] or "") + "\n\n")
            f.write("**Contextos (até 3):**\n")
            for c in row["contexts"][:3]:
                f.write(f"- { (c[:160] + '...') if len(c) > 160 else c }\n")
            f.write("\n---\n\n")

    print("[DONE] Avaliação salva em eval/report.md e eval/raw_results.csv")

if __name__ == "__main__":
    run_eval()
