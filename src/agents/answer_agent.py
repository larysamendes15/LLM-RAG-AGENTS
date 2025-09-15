from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
import os

SYSTEM = """Você é um assistente sobre a Reforma Tributária (Brasil).
Use apenas o contexto fornecido. SEM aconselhamento jurídico.
Cite as fontes após CADA sentença factual usando colchetes numerados [1], [2], ...
No final, liste 'Referências' mapeando cada número para uma URL exata do contexto.
Nunca invente links."""

TEMPLATE = """{system}

# Pergunta
{question}

# Contexto
{context}

# Instruções de citação (OBRIGATÓRIO)
- Após CADA sentença factual, inclua um marcador [n] (por ex.: "O IBS incide sobre X. [1]").
- Reutilize o mesmo [n] quando a mesma fonte sustentar mais de uma sentença.
- Ao final, escreva:

Referências
[1] URL (Título, p. X se houver)
[2] URL (Título, p. X)
...

- Só use URLs que aparecem no contexto acima.
- Se o contexto for insuficiente, responda literalmente:
"Não encontrei evidências nas fontes indexadas para responder com segurança."
"""

def _fmt_ctx(chunks):
    parts = []
    for i, c in enumerate(chunks, start=1):
        page = f"(p. {c.get('page')})" if c.get("page") is not None else ""
        snippet = (c.get("content") or "").replace("\n", " ")[:500]
        parts.append(f"[{i}] {c.get('title') or '(sem título)'} {page}\n{c.get('source')}\nTrecho: {snippet}")
    return "\n\n".join(parts)

def answer(question: str, chunks):
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL","llama3.1:8b"),
        base_url=os.getenv("OLLAMA_HOST","http://localhost:11434"),
        temperature=0.0,
        # limite para acelerar:
        num_predict=int(os.getenv("OLLAMA_NUM_PREDICT","256")),
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX","2048")),
    )
    prompt = PromptTemplate.from_template(TEMPLATE).format(
        system=SYSTEM, question=question, context=_fmt_ctx(chunks)
    )
    resp = llm.invoke(prompt)
    return {"answer": getattr(resp, "content", str(resp)), "used_chunks": chunks}
