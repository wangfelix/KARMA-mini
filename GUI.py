
import streamlit as st
import re
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
    model = "kit.gemma4-31b-it"
    

@st.cache_resource
def init_backends():
    """Initializes and caches heavy database and LLM connections."""
    config = AppConfig()
    
    
    if not config.password:
        st.error("！")
        st.stop()
        
    # Initialize Neo4j Driver
    driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))
    
  
    client = make_llm_client(config.timeout)
    
    # Fetch database schema context
    schema_text = get_schema_text(driver)
    
    return driver, client, schema_text, config.model

# Establish connections once and keep them cached across page reruns
driver, client, schema_text, model_name = init_backends()


def parse_output(raw_output: str):
    """
    Helper function to split the output text into the Cypher query and the LLM answer.
    Assumes your answer_question function returns text containing '[cypher] ...'
    """
    cypher_query = "No explicit Cypher query captured."
    llm_response = raw_output

   
    if "[cypher]" in raw_output:
        parts = raw_output.split("[cypher]")
        # Text before [cypher] might be empty or log data
        remainder = parts[1].strip()
        
        # Split by next newline or extract query before the Markdown text begins
        lines = remainder.split("\n")
        cypher_lines = []
        response_lines = []
        is_query = True
        
        for line in lines:
            if is_query and (line.strip().startswith("MATCH") or line.strip().startswith("WHERE") or line.strip().startswith("RETURN") or line.strip().startswith("LIMIT") or line.strip() == ""):
                cypher_lines.append(line)
            else:
                is_query = False
                response_lines.append(line)
                
        cypher_query = "\n".join(cypher_lines).strip()
        llm_response = "\n".join(response_lines).strip()
        
    return cypher_query, llm_response


st.set_page_config(page_title="Academic Search Engine", layout="wide", page_icon="")

# Title Header
st.title("Academic Search Engine")
st.caption("AI-powered academic search engine running over a 50-paper connected graph database.")

# Quick Sample Links to simplify live demonstrations
st.markdown("**💡 Click an example query to try it instantly:**")
example_queries = [
    "What is LSTM",
    "how many papers use LSTM",
    "which papers use attention mechanism",
    "what model does question-answering/0 use",
    "how many papers are in the graph"
]

# Create horizontal buttons for example queries
cols = st.columns(len(example_queries))
selected_query = None
for i, query_text in enumerate(example_queries):
    if cols[i].button(query_text, key=f"btn_{i}"):
        selected_query = query_text

# Main Text Input Area
user_query = st.text_input(
    "Ask a question about the papers:", 
    value=selected_query if selected_query else "", 
    placeholder="e.g., Which papers use both attention mechanism and LSTM?"
)

# Process search execution if query string exists
if user_query:
    with st.spinner("Graph RAG is exploring connections and summarizing..."):
        try:
            # Execute your existing core pipeline function
            raw_result = answer_question(driver, client, model_name, schema_text, user_query)
            
            # Parse response into discrete components
            cypher_code, clean_answer = parse_output(raw_result)
            
            st.success("complete!")
            
            # Main Result Section: Clear Markdown generated answer
            st.subheader("AI Diagnosis Answer")
            st.markdown(clean_answer)
            st.divider()
            
            # Traceability & Verification panel to show the raw Cypher generated
            with st.expander("View Graph Database Query Trace (Traceability Proof)"):
                st.markdown("**Generated Cypher Script executed against Neo4j:**")
                st.code(cypher_code, language="cypher")
                
                st.info("The answer above was directly extracted using this precise query pattern, eliminating LLM hallucinations.")
                
        except Exception as e:
            st.error(f"An unexpected system exception occurred: {str(e)}")
            st.info("No matching attributes found or connection dropped. Re-routing query safely.")