import sqlite3
from typing import Annotated, Literal, TypedDict
from langchain.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field


class Router(BaseModel):
    """Decide which worker to call next based on the tasks"""
    next: Literal["Researcher", "Writer", "FINISH"] = Field(
            description="The next worker to call. If the task is complete, use 'FINISH'"
        )


# 1. Setup LLM & Specialists
llm = ChatOllama(model="deepseek-v3.1:671b-cloud", 
                 base_url="http://localhost:11434", 
                 temperature=0.7 # Allow some variation for better tool use
                 )

# Dedicated supervisor LLM for routing (stricter, simpler)
supervisor_llm = ChatOllama(model="deepseek-v3.1:671b-cloud", 
                            base_url="http://localhost:11434",
                            format="json",
                            temperature=0)  # Zero temperature for deterministic routing

# Worker 1: Researcher (has Search tool)
search_agent = create_agent(
    model=llm,
    tools=[DuckDuckGoSearchRun()],
    system_prompt="You are a research assistant. Use search to find facts, then save important findings."
)

# Worker 2: Writer (has no tools, juist writes notes)
writer_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a note-taking assistant. Save important findings to a local file."
)


# 2. Define Supervisor with Structured Output
# Note: We don't use structured output anymore - we extract keywords from the response instead
# supervisor_node = supervisor_llm.with_structured_output(Router)

class AgentState(TypedDict):
    # 'messages' tracks the conversation history for each agent
    messages: Annotated[list[BaseMessage], lambda x, y : x + y]


# 3. Create the Node Functions
def call_supervisor(state: AgentState):
    # Node must return a dict that updates the state
    return {}

def route_to_worker(state: AgentState):
    """Separate routing function for conditional edges - uses keyword extraction instead of JSON parsing"""
    system_prompt = SystemMessage(content="""You are a supervisor routing to workers.
Given the task and responses so far, which should handle the next step?
- If information needs to be found: say "Researcher"
- If findings need to be summarized or written: say "Writer"  
- If the task is complete: say "FINISH"
Answer with ONE WORD: Researcher, Writer, or FINISH""")
    
    # Only pass a summary of the last few messages to avoid confusion
    recent_messages = state["messages"][-3:] if len(state["messages"]) > 3 else state["messages"]
    messages = [system_prompt] + recent_messages
    
    try:
        # Get raw response text (don't try to parse JSON)
        response = supervisor_llm.invoke(messages)
        response_text = response.content.upper()
        
        # Extract routing decision using keyword matching
        if "FINISH" in response_text:
            return "FINISH"
        elif "RESEARCHER" in response_text:
            return "Researcher"
        elif "WRITER" in response_text:
            return "Writer"
        else:
            # Fallback: check if both agents have been called
            agent_calls = sum(1 for msg in state["messages"] if hasattr(msg, 'name') and msg.name in ["Researcher", "Writer"])
            if agent_calls >= 2:
                return "FINISH"
            elif any(hasattr(msg, 'name') and msg.name == "Researcher" for msg in state["messages"]):
                return "Writer"
            else:
                return "Researcher"
    except Exception as e:
        # If anything fails, use sequential routing logic
        print(f"Routing error (using sequential logic): {e}")
        
        # Check if we've already called Researcher and Writer
        agent_calls = sum(1 for msg in state["messages"] if hasattr(msg, 'name') and msg.name in ["Researcher", "Writer"])
        
        # If we've called both agents, finish
        if agent_calls >= 2:
            return "FINISH"
        
        # If we haven't called Researcher yet, call it
        researcher_called = any(hasattr(msg, 'name') and msg.name == "Researcher" for msg in state["messages"])
        if not researcher_called:
            return "Researcher"
        
        # Otherwise call Writer
        return "Writer"

def run_researcher(state: AgentState) -> AgentState:
    print("--- Supervisor delegated to Researcher ---")
    response = search_agent.invoke(state)
    return {"messages": [HumanMessage(content=response["messages"][-1].content, name="Researcher")]}

def run_writer(state: AgentState) -> AgentState:
    print("--- Supervisor delegated to Writer ---")
    response = writer_agent.invoke(state)
    return {"messages": [HumanMessage(content=response["messages"][-1].content, name="Writer")]}


# 4. Build the Graph
builder = StateGraph(AgentState)

builder.add_node("Researcher", run_researcher)
builder.add_node("Writer", run_writer)
builder.add_node("Supervisor", call_supervisor) 

builder.add_edge(START, "Supervisor")

# Logic: Start at Supervisor -> Route to Worker -> Return to Supervisor (loop)
builder.add_conditional_edges(
    "Supervisor", 
    # Use the separate routing function
    route_to_worker, 
    {
        "Researcher": "Researcher",
        "Writer": "Writer",
        "FINISH": END
    }
)
# After each worker finishes, we go back to the supervisor to check if more work is needed
builder.add_edge("Researcher", "Supervisor") # Simplified for this demo
builder.add_edge("Writer", "Supervisor")


# 5. Compile with persistent memory
memory = SqliteSaver(sqlite3.connect("dynamic_team.db", check_same_thread=False))
graph = builder.compile(checkpointer=memory)


# 6. Run
config = {"configurable": {"thread_id": "manager_v1"}}
query = "Search for the price of Bitcoin and write a 1-sentence summary."

for chunk in graph.stream({"messages": [HumanMessage(content=query)]}, config):
    print(chunk)