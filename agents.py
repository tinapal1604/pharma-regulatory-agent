# What this file does:
#   Builds a LangGraph pipeline with 5 nodes:
#     1. retriever        — fetches relevant chunks from ChromaDB
#     2. clause_extractor — pulls key regulatory clauses from chunks
#     3. compliance_checker — checks clauses against ICH/FDA rules
#     4. gap_detector     — finds missing mandatory sections
#     5. aggregator       — combines all outputs into final answer
import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, END
load_dotenv()


CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
PRIMARY_LLM        = os.getenv("PRIMARY_LLM", "gemini-2.0-flash")
FALLBACK_LLM       = os.getenv("FALLBACK_LLM", "llama-3.3-70b-versatile")
RETRIEVAL_TOP_K    = int(os.getenv("RETRIEVAL_TOP_K", "5"))


class AgentState(TypedDict):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    clauses: List[str]
    compliance_issues: List[str]
    gaps: List[str]
    final_answer: str

print("Loading models and connecting to database...")

# Load embedding model (same one used in Ingest.py)
model_name = EMBEDDING_MODEL.replace("/sentence-transformers/", "")
embedding_model = SentenceTransformer(model_name)
# embedding is now ready to convert text to vectors

chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = chroma_client.get_collection("regulatory_docs")
print("Initialization complete. Agent is ready to process queries.")

print("\n Connectedd to ChromaDB at:", CHROMA_PERSIST_DIR)

# Initialize LLMs
llm = None
try:
    llm = ChatGoogleGenerativeAI(model=PRIMARY_LLM, temperature=0.0)
    llm.invoke([HumanMessage(content="Say OK")])
    print(f"Primary LLM '{PRIMARY_LLM}' is ready.")
except Exception as e:
    print(f"Error initializing primary LLM '{PRIMARY_LLM}': {e}")
    llm = ChatGroq(model=FALLBACK_LLM, temperature=0.0)
    print(f"Falling back to '{FALLBACK_LLM}'.")
print("Ready to process queries with LLM:")
    
    # Node 1 - Retriever
def retriver_node(state: AgentState) -> Dict:
    print("\n[Agent 1: Retriver] searching knownledge base....")
    query = state['query']
    query_vector = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_vector, 
        n_results=RETRIEVAL_TOP_K, 
        include=["documents", "metadatas", "distances"]
        )
    

    chunks = []
    for text, meta, dist in zip(
        results['documents'][0], 
        results['metadatas'][0], 
        results['distances'][0]
        ):
        similarity = round(1 - dist, 3)
        chunks.append({
            'text': text, 
            'source':meta.get('filename', 'unknown'),
            'page': meta.get('page', '?'),
            'similarity': similarity
            })
        print(f"  Found: {meta.get('filename','?')} p.{meta.get('page','?')} (similarity: {similarity})")
        
        return {'retrieved_chunks': chunks}


# ── NODE 2: Clause Extractor ───────────────────────────────────
def clause_extractor_node(state: AgentState) -> dict:
    """
    Reads the retrieved chunks and asks the LLM to extract
    specific regulatory clauses — exact requirements, rules,
    or obligations stated in the text.
 
    This is different from summarizing — we want the actual
    clause text so the compliance checker can evaluate it precisely.
    """
    print("\n[Agent 2: Clause Extractor] Extracting regulatory clauses...")
 
    chunks = state["retrieved_chunks"]
 
    # Build a single string of all retrieved chunk texts
    # We number them so the LLM can reference them
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n--- Chunk {i+1} (from {chunk['source']}, page {chunk['page']}) ---\n"
        context += chunk["text"] + "\n"
    # The LLM will receive all 5 chunks together as one big context string
 
    # Construct the messages to send to the LLM
    messages = [
        SystemMessage(content="""You are a regulatory affairs expert specializing 
in ICH guidelines and FDA regulations. Your job is to extract specific 
regulatory clauses, requirements, and obligations from provided text.
 
Return ONLY a JSON array of strings. Each string is one specific clause.
Example format: ["Clause 1 text here", "Clause 2 text here"]
Do not include any explanation — only the JSON array."""),
 
        HumanMessage(content=f"""Extract all specific regulatory clauses, 
requirements, and obligations from the following text.
Focus on: mandatory requirements, compliance obligations, 
documentation requirements, and procedural rules.
 
TEXT:
{context}
 
Return as JSON array of strings.""")
]
    # Send to LLM and get response
    response = llm.invoke(messages)
    # response is a LangChain AIMessage object
    # response.content is the actual text string the LLM returned
 
    # Parse the LLM's JSON response into a Python list
    try:
        # Strip markdown code fences if LLM wrapped response in ```json ... ```
        raw = response.content.strip()
        if raw.startswith("```"):
            
            raw = raw.split("\n", 1)[1]
            
            raw = raw.rsplit("```", 1)[0]
 
        clauses = json.loads(raw)
        # json.loads() converts a JSON string → Python list
 
        if not isinstance(clauses, list):
            clauses = [str(clauses)]
 
    except json.JSONDecodeError:
        clauses = [response.content.strip()]
 
    print(f"  Extracted {len(clauses)} clauses")
    for i, clause in enumerate(clauses[:3]):
        # Show first 3 clauses (slicing with [:3]) as a preview
        print(f"  {i+1}. {clause[:80]}...")
 
    return {"clauses": clauses}
 
 
# NODE 3: Compliance Checker 
def compliance_checker_node(state: AgentState) -> dict:
    """
    Takes the extracted clauses and checks them against
    known ICH/FDA compliance requirements.
 
    The LLM acts as a compliance expert — it knows ICH E3, E6,
    CTD format requirements from its training data, and checks
    whether the extracted clauses fulfill those requirements.
    """
    print("\n[Agent 3: Compliance Checker] Checking compliance...")
 
    clauses = state["clauses"]
    query   = state["query"]
 
    # Format clauses as a numbered list for the LLM
    clauses_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(clauses)])
    # "\n".join() takes a list and joins items with newlines between them
    # The list comprehension formats each clause as "1. clause text"
 
    messages = [
        SystemMessage(content="""You are an ICH/FDA regulatory compliance expert.
You review regulatory document clauses and identify compliance issues.
 
Return ONLY a JSON array of strings — each string describes one compliance issue.
If there are no issues, return an empty array: []
Do not include explanation — only the JSON array."""),
 
        HumanMessage(content=f"""Review these regulatory clauses for compliance 
with ICH E3 (Clinical Study Reports), ICH E6 (GCP), and FDA guidance.
 
User's question context: {query}
 
CLAUSES TO REVIEW:
{clauses_text}
 
Identify specific compliance issues such as:
- Missing mandatory elements
- Unclear or ambiguous requirements  
- Deviations from ICH/FDA standards
- Documentation gaps
 
Return as JSON array of strings.""")
    ]
 
    response = llm.invoke(messages)
 
    # Same JSON parsing pattern as above
    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]
        issues = json.loads(raw)
        if not isinstance(issues, list):
            issues = []
    except json.JSONDecodeError:
        issues = [response.content.strip()]
 
    print(f"  Found {len(issues)} compliance issue(s)")
    for issue in issues[:3]:
        print(f"  ⚠ {issue[:80]}...")
 
    return {"compliance_issues": issues}
 
 
# ── NODE 4: Gap Detector ───────────────────────────────────────
def gap_detector_node(state: AgentState) -> dict:
    """
    Checks whether mandatory sections required by ICH E3
    are present in the retrieved content.
 
    ICH E3 defines exactly which sections a Clinical Study
    Report must contain. This agent checks for missing ones.
    """
    print("\n[Agent 4: Gap Detector] Checking for missing sections...")
 
    chunks     = state["retrieved_chunks"]
    clauses    = state["clauses"]
 
    # Combine all chunk text into one string for the LLM to scan
    all_text = " ".join([c["text"] for c in chunks])
 
    # These are the actual mandatory sections from ICH E3 guidance
    # Hardcoded because they never change — this is the standard
    MANDATORY_ICH_E3_SECTIONS = [
        "Title Page",
        "Synopsis",
        "Table of Contents",
        "Ethics (IRB/IEC approval)",
        "Investigators and Study Administrative Structure",
        "Introduction",
        "Study Objectives",
        "Investigational Plan",
        "Study Patients",
        "Efficacy Evaluation",
        "Safety Evaluation",
        "Discussion and Overall Conclusions",
        "Reference List",
        "Adverse Events",
        "Protocol",
        "Sample Case Report Forms",
        "Statistical Methods"
    ]
 
    sections_formatted = "\n".join([f"- {s}" for s in MANDATORY_ICH_E3_SECTIONS])
 
    messages = [
        SystemMessage(content="""You are a regulatory document reviewer.
Check if mandatory ICH E3 sections are present in provided document text.
 
Return ONLY a JSON array of strings — each string is a missing section name.
If all sections are present, return: []
Do not include explanation — only the JSON array."""),
 
        HumanMessage(content=f"""Check this document text for the presence of 
these mandatory ICH E3 sections:
 
{sections_formatted}
 
DOCUMENT TEXT (excerpts):
{all_text[:3000]}
 
List any mandatory sections that appear to be MISSING or NOT ADDRESSED.
Return as JSON array of strings.""")
        # all_text[:3000] — slice to first 3000 characters to stay within
        # LLM context limits. For a full system you'd be smarter about this.
    ]
 
    response = llm.invoke(messages)
 
    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]
        gaps = json.loads(raw)
        if not isinstance(gaps, list):
            gaps = []
    except json.JSONDecodeError:
        gaps = [response.content.strip()]
 
    print(f"  Found {len(gaps)} gap(s)")
    for gap in gaps:
        print(f"  ✗ Missing: {gap}")
 
    return {"gaps": gaps}
 
 
# ── NODE 5: Aggregator ─────────────────────────────────────────
def aggregator_node(state: AgentState) -> dict:
    """
    Takes all outputs from previous agents and asks the LLM
    to combine them into one clear, well-structured answer
    for the user.
 
    This is the final step — it sees everything and writes
    the response the user will actually read.
    """
    print("\n[Agent 5: Aggregator] Building final answer...")
 
    # Read everything accumulated in state by previous agents
    query              = state["query"]
    chunks             = state["retrieved_chunks"]
    clauses            = state["clauses"]
    compliance_issues  = state["compliance_issues"]
    gaps               = state["gaps"]
 
    # Format sources so the LLM can include citations
    sources = list(set([c["source"] for c in chunks]))
    # set() removes duplicates — if 3 chunks came from E3.pdf,
    # it appears only once. list() converts set back to list.
 
    # Build a comprehensive summary for the LLM
    summary = f"""
User Question: {query}
 
Sources consulted: {', '.join(sources)}
 
Extracted Clauses ({len(clauses)} found):
{chr(10).join([f'• {c}' for c in clauses])}
 
Compliance Issues ({len(compliance_issues)} found):
{chr(10).join([f'• {i}' for i in compliance_issues]) if compliance_issues else '• None identified'}
 
Missing Sections ({len(gaps)} found):
{chr(10).join([f'• {g}' for g in gaps]) if gaps else '• None identified'}
"""
    # chr(10) is the newline character "\n" — used here because
    # we're inside an f-string and can't use backslashes directly
 
    messages = [
        SystemMessage(content="""You are a senior regulatory affairs consultant.
Synthesize the analysis below into a clear, professional response.
Structure your answer with these sections:
1. Direct Answer
2. Key Regulatory Clauses Found
3. Compliance Issues (if any)
4. Missing Sections (if any)  
5. Recommendations
Always cite which document each finding came from."""),
 
        HumanMessage(content=f"""Based on this analysis, provide a comprehensive 
answer to the user's question:
 
{summary}""")
    ]
 
    response = llm.invoke(messages)
 
    # Add a guardrail — warn if no sources were cited in the answer
    # This is a simple but effective hallucination check
    answer = response.content
    if not any(source in answer for source in sources):
        answer += "\n\n⚠ Note: Answer generated from retrieved context."
 
    print("  Final answer ready.")
 
    return {"final_answer": answer}
 
 
#  STEP 4: Build the LangGraph 
# Now we wire all the nodes together into a graph.
# This defines the ORDER and FLOW of agent execution.
 
def build_graph():
    """
    Assembles the LangGraph StateGraph.
 
    A StateGraph is a directed graph where:
    - Nodes are agent functions
    - Edges define the order of execution
    - State flows between nodes automatically
    """
 
    # Create a new graph, telling it what State looks like
    builder = StateGraph(AgentState)
    # StateGraph needs to know the state schema (AgentState)
    # so it can validate state at each step
 
    # Add each agent as a node
    # .add_node(name, function) — name is how we refer to it in edges
    builder.add_node("retriever",          retriver_node)
    builder.add_node("clause_extractor",   clause_extractor_node)
    builder.add_node("compliance_checker", compliance_checker_node)
    builder.add_node("gap_detector",       gap_detector_node)
    builder.add_node("aggregator",         aggregator_node)
 
    # Define the entry point — which node runs first
    builder.set_entry_point("retriever")
 
    # Add edges — define the flow between nodes
    # .add_edge(from, to) — after "from" finishes, run "to"
    builder.add_edge("retriever",          "clause_extractor")
    builder.add_edge("clause_extractor",   "compliance_checker")
    builder.add_edge("compliance_checker", "gap_detector")
    builder.add_edge("gap_detector",       "aggregator")
    builder.add_edge("aggregator",         END)
    # END is a special LangGraph constant meaning "stop here"
 
    # Compile the graph — this validates all edges and nodes
    # and returns a runnable object
    graph = builder.compile()
 
    return graph
 
 
# ── STEP 5: Run the pipeline ───────────────────────────────────
# This block only runs when you execute: python agents.py
# If another file imports agents.py, this block is skipped.
# That's what `if __name__ == "__main__"` means.
 
if __name__ == "__main__":
 
    print("=" * 60)
    print("  PHARMA REGULATORY AGENT — Multi-Agent Pipeline")
    print("=" * 60)
 
    # Build the graph once
    graph = build_graph()
    print("\nAgent graph built successfully.")
    print("Nodes: retriever → clause_extractor → compliance_checker → gap_detector → aggregator")
 
    # Interactive loop — keep asking questions until user types 'quit'
    print("\nType your question (or 'quit' to exit):\n")
 
    while True:
        # input() pauses execution and waits for the user to type
        query = input("Your question: ").strip()
        # .strip() removes leading/trailing whitespace
 
        if query.lower() in ["quit", "exit", "q"]:
            # .lower() converts to lowercase so "Quit", "QUIT" also work
            print("Goodbye!")
            break
            # break exits the while loop
 
        if not query:
            # If user pressed Enter without typing anything, skip
            print("Please enter a question.")
            continue
            # continue skips to the next iteration of the while loop
 
        print(f"\nProcessing: '{query}'")
        print("-" * 60)
 
        # Create the initial state — only query is set
        # All other fields start empty and get filled by agents
        initial_state: AgentState = {
            "query":             query,
            "retrieved_chunks":  [],
            "clauses":           [],
            "compliance_issues": [],
            "gaps":              [],
            "final_answer":      ""
        }
 
        try:
            # .invoke() runs the entire graph from entry point to END
            # It takes the initial state, passes it through each node,
            # and returns the final state with all fields filled in
            final_state = graph.invoke(initial_state)
 
            # Print the final answer
            print("\n" + "=" * 60)
            print("FINAL ANSWER:")
            print("=" * 60)
            print(final_state["final_answer"])
 
            # Show a brief summary of what each agent found
            print("\n" + "-" * 60)
            print("AGENT SUMMARY:")
            print(f"  Chunks retrieved:    {len(final_state['retrieved_chunks'])}")
            print(f"  Clauses extracted:   {len(final_state['clauses'])}")
            print(f"  Compliance issues:   {len(final_state['compliance_issues'])}")
            print(f"  Missing sections:    {len(final_state['gaps'])}")
            print("-" * 60)
 
        except Exception as e:
            print(f"\nERROR: {e}")
            print("Check your API keys and that ingest.py has been run.")
    print("Agent session ended.")