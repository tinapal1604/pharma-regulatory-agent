
#  Phase 4: Streamlit Web Interface

import streamlit as st

import os
import shutil
# shutil is built-in Python — used to copy uploaded files
# from Streamlit's temp folder to our docs/regulatory/ folder
import time
from dotenv import load_dotenv
from agents import build_graph, AgentState
load_dotenv()
DOCS_DIR = os.getenv("DOCS_DIR", "./docs/regulatory")


# PAGE CONFIGURATION 
st.set_page_config(
    page_title="Pharma Regulatory Agent",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)


if "graph" not in st.session_state:
    st.session_state.graph = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "graph_built" not in st.session_state:
    st.session_state.graph_built = False


#  HELPER FUNCTIONS 

def initialize_graph():
    """
    Builds the LangGraph agent pipeline.
    Called once when the app starts or when user clicks 'Initialize'.
    Stored in session_state so it persists across reruns.
    """
    with st.spinner("Loading models and connecting to database..."):
        # st.spinner() shows an animated loading indicator
        # Everything inside the `with` block runs while spinner shows
        try:
            graph = build_graph()
            st.session_state.graph = graph
            st.session_state.graph_built = True
            return True
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            return False

# We need to save uploaded PDFs to disk so our ingestion script can access them
def save_uploaded_pdf(uploaded_file):
    os.makedirs(DOCS_DIR, exist_ok=True)
    save_path = os.path.join(DOCS_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

# This function runs the full agent pipeline for a given query.
def run_agent_pipeline(query: str):
    # Create the initial state — same as in agents.py
    initial_state: AgentState = {
        "query":             query,
        "retrieved_chunks":  [],
        "clauses":           [],
        "compliance_issues": [],
        "gaps":              [],
        "final_answer":      ""
    }

    # Create UI placeholders for live agent status updates and progress bar
    status_placeholder = st.empty()

    # Show progress bar st.progress(0) shows a progress bar starting at 0%
    progress = st.progress(0)
  
    agent_names = [
        "Retriever",
        "Clause Extractor",
        "Compliance Checker",
        "Gap Detector",
        "Aggregator"
    ]

    # Show initial status
    status_placeholder.info("🔄 Starting agent pipeline...")

    try:
        start_time = time.time()

        # Run the graph — this executes all 5 agents in sequence
        with st.spinner("Agents working..."):
            final_state = st.session_state.graph.invoke(initial_state)

        elapsed = round(time.time() - start_time, 1)

        # Update progress to 100% when done
        progress.progress(100)
        status_placeholder.success(f"✅ All agents completed in {elapsed}s")

        return final_state

    except Exception as e:
        status_placeholder.error(f"❌ Pipeline error: {e}")
        progress.progress(0)
        return None


#  SIDEBAR 
with st.sidebar:
    st.title("⚕️ Pharma Agent")
    st.markdown("*Regulatory Document Analysis*")
    st.divider()

    # Initialize button 
    st.subheader("1. Initialize")

    if not st.session_state.graph_built:

        if st.button("🚀 Start Agent Pipeline", use_container_width=True):
          
            success = initialize_graph()
            if success:
                st.success("Pipeline ready!")
    else:
        st.success("✅ Pipeline active")
        if st.button("🔄 Reinitialize", use_container_width=True):
            st.session_state.graph_built = False
            st.session_state.graph = None
            st.rerun()
    st.divider()

    # PDF Upload 
    st.subheader("2. Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload regulatory PDFs",
        type=["pdf"],         
        accept_multiple_files=True,  
        help="Upload ICH guidelines, FDA guidance, or CSR documents"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = save_uploaded_pdf(uploaded_file)
            st.success(f"✅ Saved: {uploaded_file.name}")

        st.info("Re-run ingest.py in terminal to index new documents")

    # Show currently indexed documents
    st.divider()
    st.subheader("📁 Indexed Documents")

    if os.path.exists(DOCS_DIR):
        pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
        if pdf_files:
            for pdf in pdf_files:
                st.markdown(f"📄 {pdf}")
        else:
            st.warning("No PDFs found in docs/regulatory/")
    else:
        st.warning("docs/regulatory/ folder not found")

    st.divider()

    # Settings
    st.subheader("⚙️ Settings")

    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="How many document chunks to retrieve per query"
    )
    show_chunks = st.checkbox(
        "Show retrieved chunks",
        value=False,
        help="Display raw text chunks retrieved from ChromaDB"
    )

    st.divider()
    st.caption("Built with LangGraph + Gemini + ChromaDB")


#  MAIN AREA 

st.title("Pharma Regulatory Document Agent")
st.markdown(
    "Ask questions about your regulatory documents. "
    "The multi-agent pipeline retrieves relevant sections, "
    "extracts clauses, checks compliance, and identifies gaps."
)

st.divider()

#  Query Input 
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "Your question",
        placeholder="e.g. What are the GCP requirements for investigators?",
        label_visibility="collapsed"
    )

with col2:
    analyse_clicked = st.button(
        "Analyse",
        use_container_width=True,
        type="primary"
    )

#  Suggested Questions 
st.markdown("**Try these:**")

# Three columns for suggestion buttons
s1, s2, s3 = st.columns(3)

with s1:
    if st.button("GCP investigator requirements", use_container_width=True):
        query = "What are the GCP requirements for investigators?"
        analyse_clicked = True

with s2:
    if st.button("Mandatory CSR sections", use_container_width=True):
        query = "What are the mandatory sections in a clinical study report?"
        analyse_clicked = True

with s3:
    if st.button("Adverse event reporting", use_container_width=True):
        query = "What are the adverse event reporting requirements?"
        analyse_clicked = True

st.divider()

# Run Analysis when "Analyse" button is clicked and query is not empty
if analyse_clicked and query:

    if not st.session_state.graph_built:
        st.warning("⚠️ Please click 'Start Agent Pipeline' in the sidebar first.")

    else:
        # Show the query being processed
        st.markdown(f"**Query:** {query}")
        st.markdown("---")

        # Run the pipeline
        final_state = run_agent_pipeline(query)

        if final_state:

            # Agent Summary Cards
            st.subheader("Agent Summary")

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.metric(
                    label="Chunks Retrieved",
                    value=len(final_state["retrieved_chunks"])
                )

            with m2:
                st.metric(
                    label="Clauses Extracted",
                    value=len(final_state["clauses"])
                )

            with m3:
                st.metric(
                    label="Compliance Issues",
                    value=len(final_state["compliance_issues"]),
                    delta=f"-{len(final_state['compliance_issues'])} issues" if final_state["compliance_issues"] else "None",
                    delta_color="inverse"
                    # delta_color="inverse" makes positive delta red
                    # and negative delta green — opposite of default
                    # because more issues = worse, not better
                )

            with m4:
                st.metric(
                    label="Missing Sections",
                    value=len(final_state["gaps"]),
                    delta=f"-{len(final_state['gaps'])} gaps" if final_state["gaps"] else "None",
                    delta_color="inverse"
                )

            st.divider()

            #  Final Answer 
            st.subheader("Final Answer")
            st.markdown(final_state["final_answer"])

            st.divider()

            #  Expandable Detail Sections 

            with st.expander("📋 Extracted Clauses"):
                if final_state["clauses"]:
                    for i, clause in enumerate(final_state["clauses"], 1):
                        st.markdown(f"**{i}.** {clause}")
                else:
                    st.info("No clauses extracted")

            with st.expander("⚠️ Compliance Issues"):
                if final_state["compliance_issues"]:
                    for issue in final_state["compliance_issues"]:
                        st.warning(issue)
                        # st.warning() shows an orange warning box
                else:
                    st.success("No compliance issues found")

            with st.expander("✗ Missing Sections"):
                if final_state["gaps"]:
                    for gap in final_state["gaps"]:
                        st.error(f"Missing: {gap}")
                        # st.error() shows a red error box
                else:
                    st.success("No missing sections found")

            # Show raw chunks if user enabled it in settings
            if show_chunks:
                with st.expander("🔍 Retrieved Chunks (raw)"):
                    for i, chunk in enumerate(final_state["retrieved_chunks"], 1):
                        st.markdown(f"**Chunk {i}** — {chunk['source']} p.{chunk['page']} (similarity: {chunk['similarity']})")
                        st.code(chunk["text"])
                        # st.code() shows text in a monospace code block

            #  Save to History 
            st.session_state.chat_history.append({
                "query":   query,
                "answer":  final_state["final_answer"],
                "chunks":  len(final_state["retrieved_chunks"]),
                "clauses": len(final_state["clauses"]),
                "issues":  len(final_state["compliance_issues"]),
                "gaps":    len(final_state["gaps"])
            })

elif analyse_clicked and not query:
    st.warning("Please enter a question first.")


#  Chat History 
# Show previous questions at the bottom
if st.session_state.chat_history:
    st.divider()
    st.subheader("Previous Questions")

    # Show history in reverse order (newest first)
    for i, item in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Q: {item['query'][:60]}..."):
            st.markdown(item["answer"])
            # Show mini summary
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Chunks", item["chunks"])
            col_b.metric("Clauses", item["clauses"])
            col_c.metric("Issues", item["issues"])
            col_d.metric("Gaps", item["gaps"])

    # Button to clear history
    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.rerun()