from __future__ import annotations

import os
import re
from typing import Dict, Any, List

_HAS_ST = False
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_ST = True
except Exception:
    _HAS_ST = False

def _normalize(txt: str) -> str:
    txt = (txt or "").strip()
    return re.sub(r"\s+", " ", txt)

def _token_set(text: str) -> set:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    toks = re.findall(r"\w+", text, flags=re.UNICODE)
    return {t for t in toks if len(t) >= 3}

def _jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

class SelfCheckAgent:
    """
    Verifica coerência entre a resposta do LLM e até 3 chunks do contexto.
    - Padrão: similaridade Jaccard.
    - Opcional: embeddings + cosseno (SELF_CHECK_USE_EMB=1).
    Retorna uma string curta com o melhor score.
    """

    def __init__(self):
        self.ctx_clip_chars = int(os.getenv("SELF_CHECK_CTX_CHARS", "1200"))
        self.use_emb = bool(int(os.getenv("SELF_CHECK_USE_EMB", "0"))) and _HAS_ST
        default_thr = 0.55 if self.use_emb else 0.30
        self.threshold = float(os.getenv("SELF_CHECK_MIN_SIM", str(default_thr)))

        self.model = None
        if self.use_emb:
            self.model = SentenceTransformer(
                os.getenv(
                    "SELF_CHECK_EMB_MODEL",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                )
            )

    def _sim(self, a: str, b: str) -> float:
        if self.use_emb and self.model is not None:
            va = self.model.encode([a], normalize_embeddings=True)
            vb = self.model.encode([b], normalize_embeddings=True)
            return float(cosine_similarity(va, vb)[0][0])
        return _jaccard(a, b)

    def check(self, payload: Dict[str, Any]) -> str:
        """
        payload: {"question": str, "answer": str, "chunks": List[dict]}
        Usa até os 3 primeiros chunks.
        """
        answer = _normalize(payload.get("answer", ""))
        chunks: List[Dict[str, Any]] = payload.get("chunks", []) or []

        if not answer:
            return "SKIP: resposta vazia"
        if not chunks:
            return "SKIP: sem chunks"

        sims = []
        for c in chunks[:3]:
            ctx = _normalize(str(c.get("content", "")))[: self.ctx_clip_chars]
            if not ctx:
                continue
            sims.append(self._sim(answer, ctx))

        if not sims:
            return "SKIP: chunks vazios"

        best = max(sims)
        status = "OK" if best >= self.threshold else "ALERTA"
        if status == "OK":
            return f"OK: similaridade {best:.2f} ≥ {self.threshold:.2f}"
        return f"ALERTA: similaridade {best:.2f} < {self.threshold:.2f}"

    def __call__(self, payload: Dict[str, Any]) -> str:
        return self.check(payload)