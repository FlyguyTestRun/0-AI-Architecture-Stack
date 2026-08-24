"""Streamlit frontend (layer 1).

Run with:  streamlit run apps/streamlit_app/app.py

This UI deliberately exposes the internals: which backend served each layer, which
chunks were retrieved and what the agent did step by step. For an SMB pilot that
transparency is what earns trust in the answers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow running via `streamlit run` without installing the package first.
SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zerostack.app import ZerostackApp  # noqa: E402

st.set_page_config(page_title="Zerostack", page_icon="○", layout="wide")


@st.cache_resource(show_spinner="Building the stack...")
def load_app() -> ZerostackApp:
    return ZerostackApp()


app = load_app()
health = app.health()
layers = health["layers"]

with st.sidebar:
    st.subheader("Layer status")

    llm = layers["llm"]
    if llm["ollama_reachable"]:
        st.success(f"LLM: {llm['active_provider']} / {llm['model']}")
    else:
        st.warning("LLM: offline extractive provider")
        st.caption("Start Ollama with `make up` for generated answers.")

    rag = layers["rag"]
    st.info(f"RAG: {rag['vector_store']} ({rag['vector_store_mode']})")
    st.caption(
        f"embeddings: {rag['embeddings']} | dim {rag['dimensions']} | "
        f"{rag['indexed_chunks']} chunk(s) indexed"
    )
    st.info(f"Orchestrator: {layers['orchestrator']['active']}")
    st.caption(f"tools: {', '.join(layers['tools']['names']) or 'none'}")

    st.divider()
    if st.button("Ingest sample corpus", use_container_width=True):
        with st.spinner("Indexing..."):
            report = app.ingest()
        st.success(f"Indexed {report['chunks']} chunk(s) from {report['files']} file(s)")
        st.cache_resource.clear()
        st.rerun()

st.title("Zerostack")
st.caption("A zero cost AI architecture baseline. Every answer is grounded and traced.")

if rag["indexed_chunks"] == 0:
    st.warning(
        "Nothing is indexed yet. Use **Ingest sample corpus** in the sidebar, or run "
        "`zerostack ingest` to load your own documents."
    )

question = st.text_input(
    "Question",
    placeholder="How much can a support agent refund without approval?",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Running the agent..."):
        result = app.ask(question)

    st.markdown("### Answer")
    st.write(result.answer)

    columns = st.columns(4)
    columns[0].metric("Engine", result.orchestrator)
    columns[1].metric("Provider", result.llm_provider)
    columns[2].metric("Latency", f"{result.latency_ms:.0f} ms")
    columns[3].metric("Sources", len(result.sources))

    if result.sources:
        st.caption("Grounded in: " + ", ".join(result.sources))

    with st.expander("Agent steps"):
        for step in result.steps:
            st.markdown(f"**{step.name}**, {step.detail}")
            if step.data:
                st.json(step.data, expanded=False)

    with st.expander("Trace spans"):
        st.json(app.last_trace(), expanded=False)

with st.expander("Recent runs"):
    records = app.recent_runs(limit=10)
    if records:
        st.dataframe(
            [
                {
                    "when": record["created_at"][:19],
                    "question": record["question"],
                    "engine": record["orchestrator"],
                    "provider": record["llm_provider"],
                    "ms": record["latency_ms"],
                }
                for record in records
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No runs recorded yet.")
