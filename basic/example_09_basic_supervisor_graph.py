import os
import sqlite3
from typing import Annotated, TypedDict
from langchain.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph


# 1. Setup LLM & Specialists
llm = ChatOllama(model="deepseek-v3.1:671b-cloud", base_url="http://localhost:11434")


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

# 2. Define the Graph State
class AgentState(TypedDict):
    # 'messages' tracks the conversation history for each agent
    messages: Annotated[list, lambda x, y : x + y]
    # 'next' tells the supervisor who to call next
    next: str

# 3. Create the Node Functions
def call_researcher(state: AgentState) -> AgentState:
    print("--- ACTING: Researcher ---")
    response = search_agent.invoke(state)
    return {"messages": [HumanMessage(content=response["messages"][-1].content, name="Researcher")], "next": "supervisor"}

def call_writer(state: AgentState) -> AgentState:
    print("--- ACTING: Writer ---")
    response = writer_agent.invoke(state)
    return {"messages": [HumanMessage(content=response["messages"][-1].content, name="Writer")], "next": "supervisor"}

def supervisor(state: AgentState) -> AgentState:
    # The supervisor decides who goes next or if we are FINISHED
    # In a real app, you'd use a specific 'router' promt here
    last_message = state["messages"][-1].content;
    if "summary" in last_message.lower() or "write" in last_message.lower():
        return "Writer"
    elif "search" in last_message.lower() or "find" in last_message.lower():
        return "Researcher"
    return  "FINISH"

# 4. Build the Graph
builder = StateGraph(AgentState)
builder.add_node("Researcher", call_researcher)
builder.add_node("Writer", call_writer)

# Supervisor Logic (Conditional Edges)
builder.add_conditional_edges(
    START, 
    supervisor, 
    {"Researcher": "Researcher", "Writer": "Writer", "FINISH": END}
)
builder.add_edge("Researcher", END) # Simplified for this demo
builder.add_edge("Writer", END)


# Compile with persistent memory
memory = SqliteSaver(sqlite3.connect("supervisor.db", check_same_thread=False))
graph = builder.compile(checkpointer=memory)


# 6. Run
config = {"configurable": {"thread_id": "team_1"}}
inputs = {"messages": [HumanMessage(content="Search for the current price of Gold and write a short summary.")]}

for chunk in graph.stream(inputs, config):
    print(chunk)