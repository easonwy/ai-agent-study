"""
pip install langchain_ollama langchain_community pypdf chromadb langgraph
"""

import sqlite3
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2.errors import PdfReadError
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent


# --- STEP 1: INGESTION ---
# Load and chunk your local PDF
try:
    loader = PyPDFLoader("AI.pdf")
    docs = loader.load()
except PdfReadError as e:
    print(f"Error reading PDF: {e}")
    docs = []

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs) if docs else []

if not splits:
    print("No document chunks available; please check the PDF file and try again.")
    raise SystemExit(1)

# Create a local vector store using Ollama embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# --- STEP 2: DEFINE THE RETRIEVAL TOOL ---
def retrieve_docs(query: str):
    """Search for specific information in the local PDF database."""
    results = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in results])

# --- STEP 3: THE AGENTIC LOOP ---
llm = ChatOllama(model="qwen3.5:397b-cloud", base_url="http://localhost:11434", temperature=0)

# The "Agentic" part: We tell the LLM it must verify if the retrieved info actually answers the question
system_prompt = """You are a self-correcting research assistant.
1. Use the 'retrieve_docs' tool to find information.
2. If the retrieved info is irrelevant or doesn't answer the question, rewrite your search query and try again.
3. Only provide a final answer if you are confident it is grounded in the retrieved text."""

memory = SqliteSaver(sqlite3.connect("agentic_rag.db", check_same_thread=False))
agent = create_agent(
    llm,
    tools=[retrieve_docs],
    checkpointer=memory,
    system_prompt=system_prompt
)

# --- STEP 4: EXECUTION ---
config = {"configurable": {"thread_id": "rag_session_1"}}
query = "What is the main conclusion of the executive summary?"

for chunk in agent.stream({"messages": [("user", query)]}, config):
    print(chunk)