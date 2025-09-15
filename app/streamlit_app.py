import os, time, sys, pathlib, streamlit as st
from dotenv import load_dotenv
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.graph import build_graph, GraphState

load_dotenv() # ler as variaveis de ambiente env

st.set_page_config(page_title="Assistente Reforma Tributária", layout="wide") 
st.title("📘 Assistente Reforma Tributária")

q = st.text_input("Pergunta (ex: O que é IBS? )")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if st.button("Perguntar") and q.strip():
    t0 = time.time()
    try:
        with st.spinner("Consultando (retriever → answer → self-check)…"):
            out = st.session_state.graph.invoke(GraphState(question=q.strip()))

        # out é um dict
        check  = out.get("check", {})
        result = out.get("result", {})
        used   = result.get("used_chunks", [])
        ans    = result.get("answer", "")

        # Debug visível
        st.write(":memo: **Debug**",
                 {"len_used_chunks": len(used),
                  "has_answer": bool(ans),
                  "check": check})

        if check and not check.get("ok"):
            st.warning(check.get("message"))
        else:
            if not ans:
                st.info("O LLM não retornou conteúdo (answer vazio). Veja o Debug acima.")
            st.markdown(ans)

            with st.expander("Fontes utilizadas"):
                if not used:
                    st.info("Nenhum chunk foi marcado como 'usado'.")
                for i, ch in enumerate(used, start=1):
                    title = ch.get("title") or "(sem título)"
                    page  = f" (p. {ch.get('page')})" if ch.get("page") is not None else ""
                    src   = ch.get("source") or ""
                    st.markdown(f"**[{i}] {title}{page}**  \n{src}")

    except Exception as e:
        st.error("Falha ao processar a pergunta. Veja a exceção abaixo.")
        st.exception(e)

    st.caption(f"Latência: {(time.time()-t0)*1000:.0f} ms")