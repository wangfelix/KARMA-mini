import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Streamlit executes this file as a script and may only add ``graph_rag/`` to
# sys.path. Add the repository root so package imports work regardless of the
# directory from which ``streamlit run`` is invoked.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_rag.qa_neo4j import answer_question, get_schema_text, make_llm_client
from graph_rag.visualization import subgraph_to_dot, subgraph_to_html
from karma_mini.rag import Embedder, answer, hybrid_search, load_index

load_dotenv()

st.set_page_config(page_title="GraphRAG vs RAG", layout="wide", page_icon="🔎")


class AppConfig:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    api_key = os.getenv("KIT_API_KEY")
    base_url = os.getenv("KIT_BASE_URL")
    timeout = 30
    model = "kit.mistral-small-4-119b-a8b"
    rag_index = Path(os.getenv("RAG_INDEX_PATH", PROJECT_ROOT / "data" / "rag"))
    rag_top_k = int(os.getenv("RAG_TOP_K", "5"))


@st.cache_resource
def init_graph_backend():
    """Initialize and cache the GraphRAG backend.

    NOTE: schema_text is fetched once here and cached for the whole
    session -- if the underlying Neo4j database is reloaded (e.g. someone
    reruns graph_rag/load_neo4j.py) while this app is running, the cached schema
    will go stale. Restart the Streamlit app after reloading data."""
    config = AppConfig()

    missing = [
        name for name, value in [
            ("NEO4J_PASSWORD", config.password),
            ("KIT_API_KEY", config.api_key),
            ("KIT_BASE_URL", config.base_url),
        ] if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Add them to .env."
        )

    driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))
    client = make_llm_client(config.timeout)
    schema_text = get_schema_text(driver)

    return driver, client, schema_text, config.model


@st.cache_resource
def init_rag_backend():
    """Initialize and cache the plain-RAG backend."""
    config = AppConfig()
    missing = [
        name for name, value in [
            ("KIT_API_KEY", config.api_key),
            ("KIT_BASE_URL", config.base_url),
        ] if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Add them to .env."
        )
    client = make_llm_client(config.timeout)
    rag_index = load_index(str(config.rag_index))
    rag_embedder = Embedder(client, model=rag_index.meta["embed_model"])

    return client, config.model, rag_index, rag_embedder, config.rag_top_k


# Establish connections once and keep them cached across page reruns. A broken
# backend remains visible as an error without preventing the other one running.
graph_backend = rag_backend = None
graph_init_error = rag_init_error = None
try:
    graph_backend = init_graph_backend()
except Exception as exc:
    graph_init_error = exc
try:
    rag_backend = init_rag_backend()
except Exception as exc:
    rag_init_error = exc

st.title("GraphRAG vs. RAG")
if graph_init_error:
    st.warning(f"GraphRAG is currently unavailable: {graph_init_error}")
if rag_init_error:
    st.warning(
        f"Plain RAG is currently unavailable: {rag_init_error}. "
        "Build its index with `python rag.py index` if needed."
    )

example_queries = [
    "What is LSTM",
    "how many papers use LSTM",
    "which papers use attention mechanism",
    "what model does question-answering/0 use",
    "how many papers are in the graph",
]

if "question" not in st.session_state:
    st.session_state.question = ""

st.markdown(
    """
    <style>
    .st-key-example_questions div[data-testid="stButton"] > button {
        align-items: center;
        display: flex;
        height: 4.5rem;
        justify-content: center;
        line-height: 1.25;
        padding: 0.65rem 0.8rem;
        white-space: normal;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(key="example_questions"):
    st.markdown("**Try an example:**")
    cols = st.columns(len(example_queries))
    for i, query_text in enumerate(example_queries):
        if cols[i].button(query_text, key=f"btn_{i}", width="stretch"):
            st.session_state.question = query_text
            st.rerun()

with st.form("question_form"):
    user_query = st.text_input(
        "Ask a question:",
        key="question",
        placeholder="e.g., Which papers use both attention mechanism and LSTM?",
    )
    submitted = st.form_submit_button("Compare answers", type="primary")


def run_graph_rag(question):
    driver, client, schema_text, model_name = graph_backend
    return answer_question(driver, client, model_name, schema_text, question)


def run_plain_rag(question):
    client, model_name, rag_index, rag_embedder, rag_top_k = rag_backend
    hits = hybrid_search(rag_index, rag_embedder, question, k=rag_top_k)
    return {"answer": answer(client, model_name, question, hits), "hits": hits}


def render_evidence_subgraph(subgraph):
    """Render the interactive graph, with a static fallback if needed."""
    if subgraph and subgraph.get("nodes"):
        try:
            components.html(
                subgraph_to_html(subgraph),
                height=640,
                scrolling=False,
            )
        except Exception as exc:
            st.warning(
                "The interactive renderer was unavailable, so a static "
                "fallback is shown."
            )
            st.graphviz_chart(
                subgraph_to_dot(subgraph),
                width="stretch",
            )
            print(f"[interactive graph renderer failed: {exc}]")
    elif subgraph and subgraph.get("error"):
        st.info(
            "The answer was generated successfully, but its evidence "
            "subgraph could not be rendered."
        )
    else:
        st.info(
            "This query returned an aggregate or no graph entities, so "
            "there is no evidence subgraph to display."
        )


if submitted and user_query.strip():
    with st.spinner("Querying both systems..."):
        with ThreadPoolExecutor(max_workers=2) as executor:
            graph_future = (
                executor.submit(run_graph_rag, user_query.strip())
                if graph_backend else None
            )
            rag_future = (
                executor.submit(run_plain_rag, user_query.strip())
                if rag_backend else None
            )

            graph_result = rag_result = None
            graph_error, rag_error = graph_init_error, rag_init_error
            if graph_future:
                try:
                    graph_result = graph_future.result()
                except Exception as exc:
                    graph_error = exc
            if rag_future:
                try:
                    rag_result = rag_future.result()
                except Exception as exc:
                    rag_error = exc

    graph_col, rag_col = st.columns(2)

    with graph_col:
        st.subheader("GraphRAG answer")
        if graph_error:
            st.error(f"GraphRAG failed: {graph_error}")
            st.info("Check that Neo4j is running and contains the loaded graph.")
        else:
            st.markdown(graph_result["answer"])
            with st.expander("Graph query trace"):
                if graph_result["cypher"]:
                    st.code(graph_result["cypher"], language="cypher")
                else:
                    st.info("No Cypher query was executed for this question.")

    with rag_col:
        st.subheader("RAG answer")
        if rag_error:
            st.error(f"RAG failed: {rag_error}")
            st.info("Check the RAG index and embedding model configuration.")
        else:
            st.markdown(rag_result["answer"])
            with st.expander("Retrieved text passages"):
                for rank, hit in enumerate(rag_result["hits"], 1):
                    chunk = hit["chunk"]
                    st.markdown(
                        f"**{rank}. `{chunk['id']}`** — combined score "
                        f"`{hit['score']:.3f}`"
                    )
                    st.caption(chunk["text"])

    if not graph_error and graph_result:
        st.divider()
        with st.expander("GraphRAG evidence graph", expanded=True):
            render_evidence_subgraph(graph_result.get("subgraph"))
