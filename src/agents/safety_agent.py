from typing import Dict, Any

DISCLAIMER = "\n\n—\n*Aviso*: conteúdo informativo baseado em fontes oficiais (gov.br/Planalto/Câmara/Senado). Não substitui consultoria jurídica."

def apply_safety(payload: Dict[str, Any]) -> Dict[str, Any]:
    answer = payload.get("answer","") + DISCLAIMER
    payload["answer"] = answer
    return payload
