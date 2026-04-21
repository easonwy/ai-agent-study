import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Configuration
PDF_PATH = "my_docs.pdf"
DB_PATH = "./my_vector_db"
MODEL_NAME = "nomic-embed-text"

def ingest_data():
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: {PDF_PATH} not found. Please create the PDF first.")
        return

    print(f"📄 Loading {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    # 2. Split text into manageable chunks
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)

    # 3. Initialize Embeddings
    print(f"🧠 Initializing embeddings ({MODEL_NAME})...")
    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    # 4. Create and persist the VectorStore
    print(f"💾 Saving to {DB_PATH}...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_PATH,
        collection_name="local_docs"
    )
    
    print("✅ Ingestion complete! Your agent can now query this data.")

if __name__ == "__main__":
    ingest_data()