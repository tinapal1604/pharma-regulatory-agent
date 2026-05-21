# What this file does:
#   1. Reads every PDF in your docs/regulatory/ folder
#   2. Extracts the raw text from each page
#   3. Splits the text into small overlapping chunks
#   4. Converts each chunk into a vector (list of numbers)
#   5. Saves everything into ChromaDB on your laptop

# After running this file once, your knowledge base is ready.
# You don't need to run it again unless you add new PDFs.



import os          
import time        

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
load_dotenv()

DOCS_DIR         = os.getenv("DOCS_DIR", "./docs/regulatory")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "50"))


print("=" * 60)
print("  PHARMA AGENT — Document Ingestion Pipeline")
print("=" * 60)
print(f"  Docs folder   : {DOCS_DIR}")
print(f"  ChromaDB path : {CHROMA_PERSIST_DIR}")
print(f"  Chunk size    : {CHUNK_SIZE} tokens")
print(f"  Chunk overlap : {CHUNK_OVERLAP} tokens")
print(f"  Embedding model: {EMBEDDING_MODEL}")
print("=" * 60)


#  Find all PDF files in the docs/regulatory/ folder
print("\n[1/5] Scanning for PDF files...")

pdf_files = [
    os.path.join(DOCS_DIR, f)        
    for f in os.listdir(DOCS_DIR)     
    if f.lower().endswith(".pdf")  
]

if not pdf_files:
    print(f"  ERROR: No PDF files found in {DOCS_DIR}")
    print("  Make sure you copied your PDFs into docs/regulatory/")
    exit(1)  # stop the program — nothing to process

print(f"  Found {len(pdf_files)} PDF file(s):")
for f in pdf_files:
    print(f"    - {os.path.basename(f)}")  


#  Extract text from each PDF 
print("\n[2/5] Extracting text from PDFs...")

all_documents = []
# This list will hold every page from every PDF as a Document object.
# If you have 3 PDFs with 100 pages each, this list will have 300 items.

for pdf_path in pdf_files:
    filename = os.path.basename(pdf_path)
    print(f"  Reading: {filename}")

  
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["filename"] = filename
        doc.metadata["doc_type"] = "regulatory_guidance"

    all_documents.extend(documents)

    print(f"    Extracted {len(documents)} pages")

print(f"\n  Total pages extracted: {len(all_documents)}")


# Split pages into chunks 
print("\n[3/5] Splitting text into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(all_documents)

print(f"  Total chunks created: {len(chunks)}")
print(f"  Average chunk length: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")

# Show a sample chunk so you can see what we're working with
print("\n  Sample chunk (first one):")
print("  " + "-" * 40)
print(f"  Content: {chunks[0].page_content[:200]}...")
print(f"  Metadata: {chunks[0].metadata}")
print("  " + "-" * 40)


# Generate embeddings 
print("\n[4/5] Generating embeddings...")
print("  (First run downloads the model ~90MB — normal)")

model_name = EMBEDDING_MODEL.replace("sentence-transformers/", "")
embedding_model = SentenceTransformer(model_name)

texts = [chunk.page_content for chunk in chunks]

start_time = time.time()

# .encode() converts a list of strings into a 2D numpy array of vectors
# Shape: (number_of_chunks, 384)
# 384 is the vector size for all-MiniLM-L6-v2
# show_progress_bar=True shows a live progress bar in your terminal
embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True,
    batch_size=32  
)

elapsed = time.time() - start_time
print(f"\n  Generated {len(embeddings)} embeddings in {elapsed:.1f}s")
print(f"  Each embedding has {len(embeddings[0])} dimensions (numbers)")


# Store in ChromaDB
print("\n[5/5] Storing in ChromaDB...")

# PersistentClient saves the database to disk at CHROMA_PERSIST_DIR
# Next time you run ingest.py it will add to the existing database
# Next time you run the agent it will load the existing database
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# A "collection" in ChromaDB is like a table in a regular database
# get_or_create_collection: create it if it doesn't exist, open it if it does
collection = chroma_client.get_or_create_collection(
    name="regulatory_docs",
    metadata={"hnsw:space": "cosine"}
)

# ChromaDB needs:
#   ids        → unique string ID for each chunk (required)
#   documents  → the text of each chunk
#   embeddings → the vector for each chunk
#   metadatas  → dict of extra info for each chunk


ids        = [f"chunk_{i}" for i in range(len(chunks))]
documents  = [chunk.page_content for chunk in chunks]
embeddings_list = embeddings.tolist()
metadatas  = [chunk.metadata for chunk in chunks]

# ChromaDB has a limit of 5461 items per .add() call
# So we add in batches of 500 to be safe
BATCH_SIZE = 500
total = len(ids)

for i in range(0, total, BATCH_SIZE):
    # Python slicing: list[start:end]
    # When i=0: adds chunks 0-499
    # When i=500: adds chunks 500-999, etc.
    batch_end = min(i + BATCH_SIZE, total)

    collection.add(
        ids=ids[i:batch_end],
        documents=documents[i:batch_end],
        embeddings=embeddings_list[i:batch_end],
        metadatas=metadatas[i:batch_end]
    )
    print(f"  Stored chunks {i+1}–{batch_end} of {total}")

print(f"\n  ChromaDB collection '{collection.name}' now has {collection.count()} chunks")


# Quick retrieval test
print("\n[BONUS] Testing retrieval with a sample query...")

test_query = "What are the requirements for clinical study reports?"

# Embed the query using the same model
query_vector = embedding_model.encode([test_query]).tolist()

# Search ChromaDB for the 3 most similar chunks
results = collection.query(
    query_embeddings=query_vector,
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

print(f"\n  Query: '{test_query}'")
print(f"  Top 3 results:\n")

for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
)):
    similarity = 1 - dist   
    print(f"  Result {i+1} (similarity: {similarity:.3f})")
    print(f"  Source: {meta.get('filename', 'unknown')} | Page: {meta.get('page', '?')}")
    print(f"  Text: {doc[:150]}...")
    print()

print("=" * 60)
print("  Ingestion complete!")
print(f"  Your knowledge base is saved at: {CHROMA_PERSIST_DIR}")
print("  You can now move to Phase 3 — building the agents.")
print("=" * 60)
