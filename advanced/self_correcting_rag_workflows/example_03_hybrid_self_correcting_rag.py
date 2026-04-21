import os
import sqlite3
from typing import Annotated, Literal, TypedDict
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver

# --- 1. SCHEMAS FOR SELF-GRADING ---
class GradeResult(BaseModel):
    """Binary socre for completeness check on retrieved documents"""
    is_complete: Literal["yes", "no"] = Field(
        description="Are the local documents sufficient to FULLY answer the question? 'yes or 'no' only, if you need outside news or comparisons, choose 'no'."
    )

# --- 2. STATE DEFINITION ---
class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x,y : x + y]
    documents: list[str]
    needs_web_search: bool

# --- 3. VECTOR STORE SETUP (The "Retrieval" engine) ---
# Initialize embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Connect to or create a local ChromaDB
# In production, this would be pre-populated with your PDFs/Docs
db_path = "./my_vector_db"

if not os.path.exists(db_path):
    raise FileNotFoundError("Vector database not found. Please run your ingestion script first.")


vectorstore = Chroma(
    collection_name="local_docs",
    embedding_function=embeddings,
    persist_directory=db_path
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# --- 4. THE NODES ---
# Using ChatOllama for the brain and a smaller local model for the grader
llm = ChatOllama(model="qwen3.5:397b-cloud", base_url="http://localhost:11434", temperature=0, format="json")
# We use a specialized "Grader" instance
grader_llm = llm.with_structured_output(GradeResult)
web_search_tool = DuckDuckGoSearchRun()

def retrieve_node(state: ResearchState) -> ResearchState:
    print("--- 📚 RETRIEVING FROM LOCAL KNOWLEDGE ---")
    # Simulate a retrieval from your PDF/DB
    # In a real app, you'd call your VectorStore here, TODO: integrate with actual retrieval
    query = state["messages"][-1].content
    docs = retriever.invoke(query)

    # Extract text content for the state
    doc_contents = [d.page_content for d in docs]
    return {"documents": doc_contents}

def grade_node(state: ResearchState) -> ResearchState:
    print("--- 🧑‍⚖️ GRADING RETRIEVAL ---")

    if not state["documents"]:
        return {"needs_web_search": True}

    doc_content = "\n".join(state["documents"])
    user_query = state["messages"][-1].content


    # 1. Create a strict prompt, 
    # We provide the exact JSON keys expected by the GradeResult pydantic model
    # Manually inject a system prompt to ensure the model understands the schema it must follow.
    prompt = [
        SystemMessage(content="""You are a strict auditor. 
        Determine if the context provided is SUFFICIENT to answer the user question COMPLETELY.
        If the user asks for 'latest', 'current', or 'external comparisons' NOT found in the text, you MUST return {"is_complete": "no"}.
        Otherwise, if the answer is fully contained, return {"is_complete": "yes"}."""),
        HumanMessage(content=f"User Question: {user_query}\n\nContext Found: {doc_content}")
    ]
    
    # 2. Invoke
    try:
        grade = grader_llm.invoke(prompt)
        if grade.is_complete == "yes":
            print("--- ✅ LOCAL DATA IS SUFFICIENT ---")
            return {"needs_web_search": False}
        print("--- 🔍 LOCAL DATA IS NOT SUFFICIENT. TRIGGERING WEB SEARCH ---")
        return {"needs_web_search": True}
    except Exception as e:
        print(f"--- 🔍 DATA INCOMPLETE FOR FULL REQUEST. TRIGGERING WEB SEARCH ---")
        return {"needs_web_search": True}
    
def web_search_node(state: ResearchState) -> ResearchState:
    print("--- 🌐 PERFORMING WEB SEARCH ---")
    query = state["messages"][-1].content
    search_results = web_search_tool.run(query)
    return {"documents": search_results, "needs_web_search": False}

def generator_node(state: ResearchState) -> str:
    print("--- ✍️ GENERATING FINAL ANSWER ---")
    context = "\n".join(state["documents"])
    user_query = state["messages"][-1].content
    prompt = f"""You are a research assistant. 
    Using the context below (which may include internal docs and web results), answer the question.
    If information comes from the web, mention that.
    
    Question: {user_query}
    Context:
    {context}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

# --- 4. THE SUPERVISOR LOGIC ---
def decide_to_generate(state: ResearchState):
    if state["needs_web_search"]:
        return "web_search"
    else:
        return "generate"

# --- 5. BUILD THE GRAPH EXECUTION ---
builder = StateGraph(ResearchState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generator_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")

builder.add_conditional_edges(
    "grade",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)

builder.add_edge("web_search", "generate")
builder.add_edge("generate", END)

memory = SqliteSaver(sqlite3.connect("agentic_rag_final.db", check_same_thread=False))
graph = builder.compile(checkpointer=memory)

# --- 6. EXECUTION ---
def ask_agent(query_text: str):
    print(f"\n{'='*50}\nUSER QUERY: {query_text}\n{'='*50}")
    config = {"configurable": {"thread_id": "hybrid_test_session"}}
    
    inputs = {"messages": [HumanMessage(content=query_text)]}
    for chunk in graph.stream(inputs, config):
        pass # Processing nodes
    
    final_state = graph.get_state(config)
    print(f"\nFINAL RESPONSE:\n{final_state.values['messages'][-1].content}\n")

# TEST: This should trigger the Hybrid flow (Local + Web)
ask_agent("Compare QuantumVolt's energy density with the latest 2026 industry news.")
