"""
Automated Tech Blog Generator

1. Researcher Agent: Uses a search tool to find the latest facts.
2. Copywriter Agent: Takes the research and formats it into a professional post.
3. Supervisor Node: Decides if the research is sufficient or if the writing needs a rewrite


pip install langchain_ollama langgraph langchain_community duckduckgo-search langgraph-checkpoint-sqlite

"""
import sqlite3
from typing import Annotated, TypedDict, Literal
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

# ---STEP 1: MODELS & TOOLS---
llm = ChatOllama(model="qwen3.5:397b-cloud", base_url="http://localhost:11434", temperature=0)
search_tool = DuckDuckGoSearchRun()

# ---STEP 2: DEFINE THE STATE SCHEMA---
class AgentState(TypedDict):
    # This keeps track of the full conversation history
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]  # Accumulate messages
    # Internal flag to track which worker is active
    next_node: str
    # New flag for the human gate
    approved: bool

# ---STEP 3: WORKER AGENTS ---
# Specialist 1: The Fack Finder
researcher_agent = create_agent(model = llm,  tools=[search_tool])

# Specialist 2: The Copywriter
writer_agent = create_agent(model = llm, tools=[])


# ---STEP 4: NODE FUNCTIONS ---
def call_researcher(state: AgentState):
    print("--- DELEGATING TO RESEARCHER ---")
    response = researcher_agent.invoke(state)
    # We wrap the output so the Supervisor knows it came from the Researcher
    return {"messages": [HumanMessage(content=response["messages"][-1].content, name="Researcher")]}

def call_writer(state: AgentState):
    print("--- DELEGATING TO WRITER ---")
    response = writer_agent.invoke(state)
    return {"messages": [HumanMessage(content=response["messages"][-1].content, name="Writer")]}

def human_review_node(state: AgentState):
    """This node acts as the gate. We will interrupt BEFORE this runs."""
    print("--- ✅ HUMAN REVIEW COMPLETED ---")
    return {"approved": True}


# 3. Router Logic
def router(state: AgentState):
    # If we haven't researched yet, go to researcher
    if not any(isinstance(m, HumanMessage) and m.name == "Researcher" for m in state["messages"]):
        return "Researcher"
    # If researched but not approved , go to the human gate
    if not state.get("approved"):
        return "HumanGate"
    # Approved? Go to Writer
    return "Writer"


# ---STEP 6: BUILD THE GRAPH---
builder = StateGraph(AgentState)

# Add our specialized nodes
builder.add_node("Researcher", call_researcher)
builder.add_node("Writer", call_writer)
builder.add_node("HumanGate", human_review_node)
# router node that makes decisions based on state
builder.add_node("Router", lambda state: {"next_node": router(state)})

# Start into the router
builder.add_edge(START, "Router")

# routing logic from the router node
builder.add_conditional_edges(
    "Router", lambda state: state["next_node"],
    {
        "Researcher": "Researcher",
        "HumanGate": "HumanGate",
        "Writer": "Writer"
    }
)

# after each worker, loop back to router for new decision
builder.add_edge("Researcher", "Router")
builder.add_edge("HumanGate", "Router")
# writer finishes the pipeline
builder.add_edge("Writer", END)

# ---STEP 7: PERSISTENCE & RUN---
# Setup SQLite Persistence
conn = sqlite3.connect("gated_team.db", check_same_thread=False)
memory = SqliteSaver(conn)
graph = builder.compile(checkpointer=memory, interrupt_before=["HumanGate"])

# Run the system
config = {"configurable": {"thread_id": "blog_post_1"}}  # Unique session ID


def run_pipeline(task):
    # FIRST PASS: Run until research is done and it hits the breakpoint
    print(f"\nUser Goal: {task}")
    for event in graph.stream({"messages": [HumanMessage(content=task)], "approved": False}, config):
        pass # Streaming internal nodes

    # Check the state
    state = graph.get_state(config)
    if "HumanGate" in state.next:
        last_research = [m.content for m in state.values["messages"] if getattr(m, 'name', '') == "Researcher"][-1]
        print(f"\n--- 📋 RESEARCH REPORT ---\n{last_research[:500]}...")
        
        feedback = input("\nApprove research? (y/n) or type 'edit' to modify it: ")
        
        if feedback.lower() == 'y':
            # RESUME: Just continue
            for event in graph.stream(None, config):
                pass
        elif feedback.lower() == 'edit':
            new_research = input("Enter the corrected research data: ")
            # Manually update the state to "fix" the researcher's work
            graph.update_state(config, {"messages": [HumanMessage(content=new_research, name="Researcher")]})
            for event in graph.stream(None, config):
                pass
        else:
            print("Process terminated.")

    final_state = graph.get_state(config)
    print(f"\n--- 📝 FINAL BLOG POST ---\n{final_state.values['messages'][-1].content}")

# Start the team
run_pipeline("Research the impact of quantum computing on cybersecurity.")
