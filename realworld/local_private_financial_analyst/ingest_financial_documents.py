import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- Configuration ---
DOCS_PATH = "./private_finance_docs" # Drop your PDFs here
DB_DIR = "./finance_vector_db" # Vector store will be saved here
EMBED_MODEL = "nomic-embed-text" # Embedding model to use
def ingest_financial_documents():
    # Load PDFs
    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)
        print(f"Created directory {DOCS_PATH}. Please place your PDF documents here.")
        return
    
    all_docs = []
    for doc in os.listdir(DOCS_PATH):
        if doc.endswith(".pdf"):
            print(f"Processing {doc}...")
            loader = PyPDFLoader(os.path.join(DOCS_PATH, doc))
            all_docs.extend(loader.load())

    if not all_docs:
        print("No documents found in the specified directory.")
        return
    
    # Chunking(Finacial docs need small chunks for precision)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    splits = splitter.split_documents(all_docs)
    print(f"Split {len(all_docs)} documents into {len(splits)} chunks.")

    # Embed & Store
    print(f"Embedding {len(splits)} chunks...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="finance_docs"
    )   
    print(f"✅ Ingested {len(splits)} chunks into the local Vector DB.")

if __name__ == "__main__":
    ingest_financial_documents()
