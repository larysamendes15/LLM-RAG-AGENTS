import re
from typing import Dict, Any

CITATION_PAT = re.compile(r"https?://\S+")

def self_check(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Regra simples: exige pelo menos 1 URL de citação na resposta.
    Futuro: validar que cada sentença assertiva tem uma citação próxima."""
    answer = payload.get("answer","")
    if not CITATION_PAT.search(answer):
        return {
            "ok": False,
            "message": "Resposta sem evidências suficientes. Não encontrei citações nas fontes indexadas."
        }
    return {"ok": True, "message": "OK"}
