import streamlit as st
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

from qa_neo4j import answer_question, get_schema_text, make_llm_client

load_dotenv()


class AppConfig:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    api_key = os.getenv("KIT_API_KEY")
    base_url = os.getenv("KIT_BASE_URL")
    timeout = 30
    model = "kit.mistral-small-4-119b-a8b"


@st.cache_resource
def init_backends():
    """Initializes and caches heavy database and LLM connections.
    NOTE: schema_text is fetched once here and cached for the whole
    session -- if the underlying Neo4j database is reloaded (e.g. someone
    reruns load_neo4j.py) while this app is running, the cached schema
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
        st.error(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            f"Copy .env.example to .env and fill in your own values."
        )
        st.stop()

    driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))
    client = make_llm_client(config.timeout)
    schema_text = get_schema_text(driver)

    return driver, client, schema_text, config.model


# Establish connections once and keep them cached across page reruns
driver, client, schema_text, model_name = init_backends()

st.set_page_config(page_title="Academic Search Engine", layout="wide", page_icon="")

st.title("Academic Search Engine")
st.caption("AI-powered academic search engine running over a 50-paper connected graph database.")

st.markdown("**Click an example query to try it instantly:**")
example_queries = [
    "What is LSTM",
    "how many papers use LSTM",
    "which papers use attention mechanism",
    "what model does question-answering/0 use",
    "how many papers are in the graph",
]

cols = st.columns(len(example_queries))
selected_query = None
for i, query_text in enumerate(example_queries):
    if cols[i].button(query_text, key=f"btn_{i}"):
        selected_query = query_text

user_query = st.text_input(
    "Ask a question about the papers:",
    value=selected_query if selected_query else "",
    placeholder="e.g., Which papers use both attention mechanism and LSTM?",
)

if user_query:
    with st.spinner("Graph RAG is exploring connections and summarizing..."):
        try:
            result = answer_question(driver, client, model_name, schema_text, user_query)

            st.success("Done!")

            st.subheader("Answer")
            st.markdown(result["answer"])
            st.divider()

            with st.expander("View Graph Database Query Trace (Traceability Proof)"):
                if result["cypher"]:
                    st.markdown("**Generated Cypher query executed against Neo4j:**")
                    st.code(result["cypher"], language="cypher")
                    st.info(
                        "The answer above was directly extracted using this precise "
                        "query pattern, grounded in the actual graph data -- not "
                        "generated freely by the LLM."
                    )
                else:
                    st.info(
                        "No Cypher query was executed for this question -- the "
                        "system determined it was out of scope for this graph."
                    )

        except Exception as e:
            st.error(f"An unexpected system exception occurred: {str(e)}")
            st.info("Try rephrasing your question, or check that Neo4j is running.")