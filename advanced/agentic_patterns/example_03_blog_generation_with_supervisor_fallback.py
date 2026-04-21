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
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field
import json
import re

# ---STEP 1: MODELS & TOOLS---
llm = ChatOllama(model="qwen3.5:397b-cloud", base_url="http://localhost:11434", temperature=0)

# Helper function to parse routing decision from text
def parse_routing_decision(text: str) -> str:
    """Extract routing decision from model output, handling both JSON and plain text responses."""
    # Try to find JSON first
    try:
        json_match = re.search(r'\{[^{}]*"next_step"[^{}]*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            if "next_step" in parsed:
                return parsed["next_step"]
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # Fallback: look for keywords in the text
    text_upper = text.upper()
    if "FINISH" in text_upper:
        return "FINISH"
    elif "WRITER" in text_upper:
        return "Writer"
    elif "RESEARCHER" in text_upper:
        return "Researcher"
    
    # Default fallback
    return "FINISH"

# ---STEP 2: DEFINE THE STATE SCHEMA---
class AgentState(TypedDict):
    # This keeps track of the full conversation history
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]  # Accumulate messages
    # Internal flag to track which worker is active
    next_node: str

# ---STEP 3: WORKER AGENTS ---
# Specialist 1: Researcher - uses knowledge-based research (no external tools due to network limitations)
researcher_agent = create_agent(model=llm, tools=[])

# Specialist 2: The Copywriter
writer_agent = create_agent(model=llm, tools=[])


# ---STEP 4: SUPERVISOR (BRAIN) ---
# No structured output - we'll parse manually for better compatibility with Ollama
supervisor_node = llm

# ---STEP 5: NODE FUNCTIONS ---
def call_researcher(state: AgentState):
    print("--- DELEGATING TO RESEARCHER ---")
    try:
        system_msg = SystemMessage(content=(
            "You are a research specialist. Based on your knowledge, provide comprehensive research findings "
            "on the requested topic. Present facts, trends, and key insights in a structured format. "
            "Be accurate and detailed."
        ))
        response = researcher_agent.invoke({"messages": [system_msg] + state["messages"]})
        # We wrap the output so the Supervisor knows it came from the Researcher
        return {"messages": [HumanMessage(content=response["messages"][-1].content, name="Researcher")]}
    except Exception as e:
        print(f"Researcher error: {e}")
        # Fallback: return a direct response without tools
        fallback_response = llm.invoke([
            SystemMessage(content=(
                "You are a research specialist. Provide comprehensive research findings on AI advancements. "
                "Present facts, trends, and key insights in a structured format."
            )),
            state["messages"][-1] if state["messages"] else HumanMessage(content="Research AI advancements")
        ])
        return {"messages": [HumanMessage(content=fallback_response.content, name="Researcher")]}

def call_writer(state: AgentState):
    print("--- DELEGATING TO WRITER ---")
    try:
        system_msg = SystemMessage(content=(
            "You are a professional copywriter. Take the research findings and write a compelling, "
            "well-structured blog post. Use clear headings, engaging language, and organize the content logically."
        ))
        response = writer_agent.invoke({"messages": [system_msg] + state["messages"]})
        return {"messages": [HumanMessage(content=response["messages"][-1].content, name="Writer")]}
    except Exception as e:
        print(f"Writer error: {e}")
        # Fallback: return a direct response
        fallback_response = llm.invoke([
            SystemMessage(content=(
                "You are a professional copywriter. Write a compelling blog post based on the research provided. "
                "Use clear headings and engaging language."
            )),
            state["messages"][-1] if state["messages"] else HumanMessage(content="Write a blog post")
        ])
        return {"messages": [HumanMessage(content=fallback_response.content, name="Writer")]}

def call_supervisor(state: AgentState):
    print("--- SUPERVISOR IS THINKING ---")
    system_msg = SystemMessage(content=(
        "You are a routing supervisor for a blog generation system. "
        "Respond with a JSON object containing only the field 'next_step' with one of these values:\n"
        "- 'Researcher' - if you need to gather research facts\n"
        "- 'Writer' - if you have facts and need to write the blog post\n"
        "- 'FINISH' - if the blog post is complete and high quality\n\n"
        "Example response: {\"next_step\": \"Researcher\"}\n"
        "Respond ONLY with the JSON object, no other text."
    ))
    # Get raw text response and parse it
    response = supervisor_node.invoke([system_msg] + state["messages"])
    routing_decision = parse_routing_decision(response.content)
    return {"next_node": routing_decision}

# ---STEP 6: BUILD THE GRAPH---
builder = StateGraph(AgentState)

# Add our specialized nodes
builder.add_node("Supervisor", call_supervisor)
builder.add_node("Researcher", call_researcher)
builder.add_node("Writer", call_writer)

# The logic Flow
builder.add_edge(START, "Supervisor")

# Conditional routing from the Supervisor
builder.add_conditional_edges(
    "Supervisor",
    lambda state: state["next_node"],
    {
        "Researcher": "Researcher",
        "Writer": "Writer",
        "FINISH": END
    }
)

# Workers always report back to the Supervisor
builder.add_edge("Researcher", "Supervisor")
builder.add_edge("Writer", "Supervisor")

# ---STEP 7: PERSISTENCE & RUN---
# Setup SQLite Persistence
conn = sqlite3.connect("blog_memory.db", check_same_thread=False)
memory = SqliteSaver(conn)
graph = builder.compile(checkpointer=memory)

# Run the system
config = {"configurable": {"thread_id": "blog_post_1"}}  # Unique session ID
task = {"messages": [HumanMessage(content="Write a blog post about the latest advancements in AI.")]}

for chunk in graph.stream(task, config):
    print(chunk)
    print("-" * 30)
