import re
from typing import Dict, Any

#valida se há URL por sentença

URL = r"https?://\S+"
SENT = re.compile(r"(?<=[\.!?])\s+")

def self_check(payload: Dict[str, Any]) -> Dict[str, Any]:
    txt = (payload.get("answer") or "").strip()
    if not txt:
        return {"ok": False, "message": "Sem resposta."}
    bad = []
    for s in [s for s in SENT.split(txt) if s.strip()]:
        if "não encontrei evidências" in s.lower():
            continue
        if not re.search(URL, s):
            bad.append(s[:120])
    if bad:
        return {"ok": False, "message": f"Faltam citações em {len(bad)} sentença(s)."}
    return {"ok": True, "message": "OK"}
