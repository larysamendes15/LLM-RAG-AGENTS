from __future__ import annotations
import re
from typing import List, Dict

_SENT_SPLIT = re.compile(r"[.!?]\s+")

FALLBACK_NO_CONTEXT = "Não foi encontrado contexto para a pergunta solicitada."

def _strip_disclaimer(text: str) -> str:
    """
    Remove o disclaimer se já tiver sido anexado ao final do texto.
    Regra simples: tudo antes de uma linha que começa com '—' (travessão).
    """
    if not text:
        return ""
    parts = text.split("\n—\n", 1)  # usa o separador que você está usando no disclaimer
    return parts[0].strip()

def fast_self_check(answer: str, allowed_sources: List[str]) -> Dict:
    """
    Modo best:
    - Se a resposta for o fallback, está OK (sem referência).
    - Caso contrário: checa concisão (≤ 4 frases). Não bloqueia por allowed_sources vazio.
    """
    txt = (answer or "").strip()
    # ignore o disclaimer na checagem
    core = _strip_disclaimer(txt)

    if core == FALLBACK_NO_CONTEXT:
        return {"ok": True, "message": "Sem contexto — resposta padrão exibida."}

    sents = [s for s in _SENT_SPLIT.split(core) if s.strip()]
    if len(sents) > 4:
        return {"ok": False, "message": "Resposta longa demais (máx. 4 frases)."}

    # NÃO bloqueie por allowed_sources vazio — isso não deve impedir a resposta
    return {"ok": True, "message": "OK"}

def apply_safety(result: dict) -> dict:
    """
    Adiciona o disclaimer APENAS se NÃO for fallback.
    Deixe para anexar o disclaimer DEPOIS do self-check (exibição/UX).
    """
    ans = (result.get("answer") or "").strip()
    if ans and ans != FALLBACK_NO_CONTEXT:
        disclaimer = (
            "\n\n—\n"
            "*Aviso*: conteúdo informativo com base em fontes oficiais "
            "(gov.br/Planalto/Câmara/Senado). Não substitui consultoria jurídica."
        )
        result["answer"] = ans + disclaimer
    return result
