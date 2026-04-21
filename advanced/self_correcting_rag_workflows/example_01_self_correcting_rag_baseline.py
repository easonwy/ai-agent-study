from typing import Annotated, Literal, TypedDict
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# --- 1. SCHEMAS FOR SELF-GRADING ---
class GradeResult(BaseModel):
    """Binary socre for relevance check on retrieved documents"""
    binary_score: Literal["yes", "no"] = Field(description="Is the retrieved document relevant to the user's question? 'yes or 'no' only.") 

# --- 2. STATE DEFINITION ---
class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x,y : x + y]
    documents: list[str]
    needs_web_search: bool

# --- 3. THE NODES ---
llm = ChatOllama(model="qwen3.5:397b-cloud", base_url="http://localhost:11434", temperature=0, format="json")
# We use a specialized "Grader" instance
grader_llm = llm.with_structured_output(GradeResult)
web_search_tool = DuckDuckGoSearchRun()

def retrieve_node(state: ResearchState) -> ResearchState:
    print("--- 📚 RETRIEVING FROM LOCAL KNOWLEDGE ---")
    # Simulate a retrieval from your PDF/DB
    # In a real app, you'd call your VectorStore here, TODO: integrate with actual retrieval
    query = state["messages"][-1].content
    return {"documents": ["Local context about Quantum Computing..."], "needs_web_search": False}

def grade_node(state: ResearchState) -> ResearchState:
    print("--- 🧑‍⚖️ GRADING RETRIEVAL ---")
    doc_content = "\n".join(state["documents"])
    user_query = state["messages"][-1].content


    # 1. Create a strict prompt, 
    # We provide the exact JSON keys expected by the GradeResult pydantic model
    # Manually inject a system prompt to ensure the model understands the schema it must follow.
    prompt = [
        SystemMessage(content="""You are a grader. 
        Evaluate if the document is relevant to the user query.
        You MUST respond in JSON format with a single key 'binary_score' and value 'yes' or 'no'.
        Example: {"binary_score": "yes"}"""),
        HumanMessage(content=f"Query: {user_query}\nDocument: {doc_content}")
    ]
    
    # 2. Invoke
    try:
        grade = grader_llm.invoke(prompt)
        
        # Handle cases where it returns a dict instead of Pydantic object
        score = grade.binary_score if hasattr(grade, 'binary_score') else grade['binary_score']
        
        if score == "yes":
            print("--- ✅ RELEVANT DATA FOUND ---")
            return {"needs_web_search": False}
        else:
            print("--- ❌ DATA IRRELEVANT. TRIGGERING WEB SEARCH ---")
            return {"needs_web_search": True}
            
    except Exception as e:
        print(f"--- ⚠️ GRADING FAILED, FALLBACK TO WEB SEARCH ---")
        return {"needs_web_search": True}
    
def web_search_node(state: ResearchState) -> ResearchState:
    print("--- 🌐 PERFORMING WEB SEARCH ---")
    query = state["messages"][-1].content
    search_results = web_search_tool.run(query)
    return {"documents": search_results, "needs_web_search": False}

def generator_node(state: ResearchState) -> str:
    print("--- ✍️ GENERATING FINAL ANSWER ---")
    context = "\n".join(state["documents"])
    prompt = f"Answer the user based on this context: {context}"
    response = llm.invoke(prompt)
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

# Compile
graph = builder.compile()

# --- 6. RUN ---
inputs = {"messages": [HumanMessage(content="Explain the latest breakthrough in Room Temperature Superconductors.")]}
for output in graph.stream(inputs):
    print(output)
