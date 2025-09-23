from __future__ import annotations
from typing import Any, Dict, List
import re

class SafetyAgent:
    """
    - Mantém a interface atual: lê 'final_answer' e 'citations' do state
    - Adiciona disclaimers temáticos (legal/saúde/finanças) uma única vez
    - Bloqueia consultas perigosas
    - Mantém 📚 Fontes quando existir
    """

    _HEALTH  = re.compile(r"\b(saúde|sintoma|diagn[oó]st|tratament|rem[eé]dio|doen[cç]a|exame|receita m[eé]dica)\b", re.I)
    _LEGAL   = re.compile(r"\b(lei|art\.?|artigo|jur[ií]dic|advog|processo|penal|civil|tribut[aá]rio|imposto|al[ií]quota|reforma tribut[aá]ria)\b", re.I)
    _FIN     = re.compile(r"\b(invest|rentabil|renda fixa|ações|derivativos|cripto|fundo|ibovespa)\b", re.I)
    _DANGER  = re.compile(r"\b(explosiv|bomba|detonador|malware|ransomware|ddos|botnet|arrombar|lockpick|suic[ií]d|auto[-\s]?les[aã]o|fabricar arma)\b", re.I)

    def __init__(self):
        self.default_disclaimer = "⚠️ Esta resposta é apenas informativa e educacional, consulte sempre materiais oficiais."

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        answer: str = (state.get("final_answer") or "").strip()
        citations: List[str] = state.get("citations", [])
        query: str = state.get("query", "")

        if not answer:
            return {
                "final_answer": "",
                "safety_applied": True,
                "agent_logs": state.get("agent_logs", []) + ["[Safety] resposta vazia — nada a aplicar"],
                "next_agent": "end",
            }

        if self._is_blocked(query, answer):
            blocked_msg = "Não posso ajudar com esse tipo de solicitação."
            return {
                "final_answer": blocked_msg,
                "safety_applied": True,
                "agent_logs": state.get("agent_logs", []) + ["[Safety] bloqueado por política (perigoso)"],
                "next_agent": "end",
            }

        final_answer = self._format_final_answer(query, answer, citations)
        log = f"[Safety] formatação aplicada: {len(citations)} citações; disclaimer contextual"

        return {
            "final_answer": final_answer,
            "safety_applied": True,
            "agent_logs": state.get("agent_logs", []) + [log],
            "next_agent": "end",
        }

    @staticmethod
    def _strip_disclaimer(text: str) -> str:
        if not text:
            return ""
        parts = text.split("\n—\n", 1)
        return parts[0].strip()

    def _is_blocked(self, question: str, answer: str) -> bool:
        text = f"{question or ''} {answer or ''}"
        return bool(self._DANGER.search(text))

    def _disclaimer_for(self, text: str) -> str:
        has_health = bool(self._HEALTH.search(text))
        has_legal  = bool(self._LEGAL.search(text))
        has_fin    = bool(self._FIN.search(text))

        lines: List[str] = []
        if has_health:
            lines.append("*Aviso*: não forneço aconselhamento médico. Procure um profissional de saúde.")
        if has_legal:
            lines.append("*Aviso*: conteúdo informativo; não substitui consultoria jurídica ou fiscal.")
        if has_fin:
            lines.append("*Aviso*: não é recomendação de investimento. Faça sua própria análise.")
        if not lines:
            lines.append(self.default_disclaimer)
        return "\n".join(lines)

    def _format_final_answer(self, query: str, answer: str, citations: List[str]) -> str:
        core = self._strip_disclaimer(answer)

        if citations and "📚 Fontes:" not in core:
            core += "\n\n📚 Fontes:\n" + "\n".join(citations)

        disclaimer = self._disclaimer_for(f"{query} {core}")
        core += f"\n\n—\n{disclaimer}"
        return core
