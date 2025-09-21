from __future__ import annotations

import os
from typing import Dict

from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

FALLBACK_NO_CONTEXT = "Não foi encontrado contexto para a pergunta solicitada."

# ----------------------------
# LLM (Ollama) — igual ao seu
# ----------------------------
def _build_llm() -> ChatOllama:
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    num_pred = int(os.getenv("OLLAMA_NUM_PREDICT", "192"))
    temp = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    return ChatOllama(base_url=base_url, model=model, num_ctx=num_ctx, num_predict=num_pred, temperature=temp)

# ----------------------------
# PROMPT — versão aprimorada
# ----------------------------
ANSWER_PROMPT_BEST = PromptTemplate.from_template(
    """
Você é um assistente sobre a reforma tributaria conciso. Use SOMENTE as informações do CONTEXTO abaixo.
Responda em português, em um único parágrafo, tente retornar pelo menos 2 linhas.
Não invente informação.

CONTEXTO:
{context}

PERGUNTA: {question}
"""
)

# ----------------------------
# Helpers de contexto/saída
# ----------------------------
def _clip(txt: str, n: int = 1000) -> str:
    return txt if len(txt) <= n else txt[:n] + "..."

def _fmt_single_ctx(chunk: Dict) -> str:
    # Sem [1] no corpo; a referência fica para a UI (se houver resposta)
    return _clip(chunk.get("content", "") or "")

def _as_text(resp) -> str:
    return getattr(resp, "content", str(resp)).strip()

# ----------------------------
# Resposta (modo único: best)
# ----------------------------
def _clip(txt: str, n: int = 1000) -> str:
    return txt if len(txt) <= n else txt[:n] + "..."

def _fmt_single_ctx(chunk: Dict) -> str:
    return _clip(chunk.get("content", "") or "")

def _as_text(resp) -> str:
    return getattr(resp, "content", str(resp)).strip()

def answer_best(question: str, top_chunk: Dict) -> str:
    # Se não houver chunk (ou vier vazio), aí sim devolvemos o fallback fixo
    if not top_chunk or not (top_chunk.get("content") or "").strip():
        return FALLBACK_NO_CONTEXT

    llm = _build_llm()
    prompt = ANSWER_PROMPT_BEST.format(
        context=_fmt_single_ctx(top_chunk),
        question=question,
    )
    text = _as_text(llm.invoke(prompt))

    # Se o modelo, por qualquer motivo, retornar string vazia, caímos para um resumo curtinho do próprio chunk
    if not text:
        text = _clip(top_chunk.get("content", "").strip(), 420)
    return text