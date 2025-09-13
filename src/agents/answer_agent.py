from typing import List, Dict, Any
import os
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate

SYSTEM_PROMPT = """Você é um assistente de RAG sobre Reforma Tributária (Brasil).
Responda de forma informativa e objetiva, SEM dar aconselhamento jurídico.
Use SOMENTE as fontes fornecidas no contexto.
Inclua CITAÇÕES obrigatórias (URL + se houver, página) após as frases assertivas.
Se não houver evidências suficientes nos trechos, diga explicitamente que não encontrou nas fontes indexadas.
"""

TEMPLATE = """{system}

# Pergunta
{question}

# Contexto (trechos recuperados)
{context}

# Instruções
- Cite a fonte após cada afirmação factual importante, no formato [Título/Órgão — URL (p. X)]
- Se não houver contexto suficiente, responda: "Não encontrei evidências nas fontes indexadas para responder com segurança."
- Não faça diagnóstico legal. Seja informativo.
"""

def format_context(chunks: List[Dict[str, Any]]) -> str:
    formatted = []
    for i, c in enumerate(chunks, start=1):
        src = c.get("source","")
        title = c.get("title") or "(sem título)"
        page = c.get("page")
        page_str = f" (p. {page})" if page is not None else ""
        formatted.append(f"[{i}] {title}{page_str}\n{src}\nTrecho:\n{c.get('content','')[:1200]}")
    return "\n\n".join(formatted)

def answer(question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    llm = ChatOllama(model=model)
    ctx = format_context(chunks)
    prompt = PromptTemplate.from_template(TEMPLATE).format(system=SYSTEM_PROMPT, question=question, context=ctx)
    resp = llm.invoke(prompt)
    return {"answer": resp.content, "used_chunks": chunks}
