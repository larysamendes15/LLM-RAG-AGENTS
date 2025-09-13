from typing import Dict, Any

def supervisor_route(question: str) -> str:
    """Router simples: por ora, manda tudo para 'rag'.
    Futuro: classificar intents (definições, benefícios, cronograma...)."""
    return "rag"
