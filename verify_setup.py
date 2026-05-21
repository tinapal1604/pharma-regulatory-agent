import sys, os
print('='*55)
print('  Pharma Agent - setup verification')
print('='*55)
errors = []

print('\n[1/5] Checking .env file...')
try:
    from dotenv import load_dotenv
    if not os.path.exists('.env'):
        errors.append('.env file not found - copy .env.example to .env and fill keys')
    else:
        load_dotenv()
        print('      .env found and loaded')
except ImportError:
    errors.append('python-dotenv not installed')

print('\n[2/5] Checking package imports...')
packages = [
    ('langgraph','langgraph'),('langchain','langchain'),
    ('langchain_google_genai','langchain-google-genai'),
    ('langchain_groq','langchain-groq'),('chromadb','chromadb'),
    ('sentence_transformers','sentence-transformers'),
    ('fitz','pymupdf'),('pdfplumber','pdfplumber'),
    ('streamlit','streamlit'),('mlflow','mlflow'),
]
for module, pip_name in packages:
    try:
        __import__(module)
        print(f'      OK  {pip_name}')
    except ImportError:
        errors.append(f'{pip_name} not installed')
        print(f'      MISSING  {pip_name}')

print('\n[3/5] Checking Google Gemini API key...')
key = os.getenv('GOOGLE_API_KEY','')
if not key or key == 'your_google_api_key_here':
    errors.append('GOOGLE_API_KEY not set - get it at aistudio.google.com')
    print('      NOT SET')
else:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)
        resp = llm.invoke('Reply with exactly: OK')
        print(f'      Connected: {resp.content.strip()[:30]}')
    except Exception as e:
        errors.append(f'Gemini error: {e}')
        print(f'      ERROR: {e}')

print('\n[4/5] Checking Groq API key...')
key = os.getenv('GROQ_API_KEY','')
if not key or key == 'your_groq_api_key_here':
    errors.append('GROQ_API_KEY not set - get it at console.groq.com')
    print('      NOT SET')
else:
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)
        resp = llm.invoke('Reply with exactly: OK')
        print(f'      Connected: {resp.content.strip()[:30]}')
    except Exception as e:
        errors.append(f'Groq error: {e}')
        print(f'      ERROR: {e}')

print('\n[5/5] Checking ChromaDB + embeddings...')
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    client = chromadb.Client()
    col = client.get_or_create_collection('test')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    emb = model.encode(['test']).tolist()
    col.add(documents=['test'], embeddings=emb, ids=['t1'])
    results = col.query(query_embeddings=emb, n_results=1)
    assert results['documents'][0][0] == 'test'
    print('      ChromaDB  OK')
    print('      Embeddings OK (all-MiniLM-L6-v2)')
    client.delete_collection('test')
except Exception as e:
    errors.append(f'ChromaDB/embeddings error: {e}')
    print(f'      ERROR: {e}')

print('\n' + '='*55)
if not errors:
    print('  All checks passed. Ready to build!')
else:
    print(f'  {len(errors)} issue(s) to fix:')
    for i,e in enumerate(errors,1):
        print(f'  {i}. {e}')
print('='*55)
