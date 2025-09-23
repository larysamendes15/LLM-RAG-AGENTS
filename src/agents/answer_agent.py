from __future__ import annotations

import os
from typing import Dict, List
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

FALLBACK_NO_CONTEXT = "Não foi encontrado contexto para a pergunta solicitada."

load_dotenv()
def _build_llm() -> ChatNVIDIA:
    model = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")  
    api_key = os.getenv("NVIDIA_API_KEY")
    temperature = float(os.getenv("NVIDIA_TEMPERATURE", "0.1"))
    max_tokens = int(os.getenv("NVIDIA_MAX_TOKENS", "512"))

    if not api_key:
        raise ValueError("Defina a variável de ambiente NVIDIA_API_KEY")

    return ChatNVIDIA(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

ANSWER_PROMPT_BEST = PromptTemplate.from_template(
    """
Você é um assistente especializado em Reforma Tributária.
Responda em português, de forma clara e completa, mas sem ultrapassar 4 a 6 frases.
Evite terminar no meio de uma ideia. Se precisar resumir, faça isso de forma natural no final.

CONTEXTO:
{context}

PERGUNTA: {question}
"""
)

def _clip(txt: str, n: int = 1000) -> str:
    return txt if len(txt) <= n else txt[:n] + "..."

def _fmt_single_ctx(chunk: Dict) -> str:
    return _clip(chunk.get("content", "") or "")

def _as_text(resp) -> str:
    return getattr(resp, "content", str(resp)).strip()


def _clip(txt: str, n: int = 1000) -> str:
    return txt if len(txt) <= n else txt[:n] + "..."

def _fmt_single_ctx(chunks: List[Dict]) -> str:
    return "\n\n".join(list(map(lambda chunk: chunk.get("content", ""), chunks)))

def _as_text(resp) -> str:
    return getattr(resp, "content", str(resp)).strip()

def answer_best(question: str, top_chunks: Dict) -> str:
    # Se não houver chunk (ou vier vazio), aí sim devolvemos o fallback fixo
    if len(top_chunks) == 0 or all((x.get("content") or "").strip() == "" for x in top_chunks):
        return FALLBACK_NO_CONTEXT

    llm = _build_llm()
    prompt = ANSWER_PROMPT_BEST.format(
        context=_fmt_single_ctx(top_chunks),
        question=question,
    )
    text = _as_text(llm.invoke(prompt))

    # Se o modelo, por qualquer motivo, retornar string vazia, caímos para um resumo curtinho do próprio chunk
    if not text:
        text = _clip(top_chunks[0].get("content", "").strip(), 420)
    return text