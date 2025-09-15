# adiciona o disclaimer se passou no self-check

def apply_safety(result: dict) -> dict:
    disclaimer = "\n\n—\n*Aviso*: conteúdo informativo com base em fontes oficiais (gov.br/Planalto/Câmara/Senado). Não substitui consultoria jurídica."
    result["answer"] = (result.get("answer") or "") + disclaimer
    return result
