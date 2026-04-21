import os
import sqlite3
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool


# 1. Setup - Use llama3.1 for reliable multi-tool coordination
llm = ChatOllama(model="deepseek-v3.1:671b-cloud", base_url="http://localhost:11434")


# 2. Define Custom Tools
@tool
def save_note(content: str) -> str:
    """Saves a note to a local file."""
    with open("research_notes.txt", "a") as f:
        f.write(f"--- Note ---\n{content}\n\n")
    return "Note saved successfully."

# Initialize built-in search tool
search_tool = DuckDuckGoSearchRun()

# 3. Persisting Memory with SQLite
conn = sqlite3.connect("memory.db", check_same_thread=False)
memory = SqliteSaver(conn)


# 4. Create the Agent
tools = [search_tool, save_note]

agent = create_agent(
    model=llm,
    tools=tools, 
    checkpointer=memory,
    system_prompt="You are a research assistant. Use search to find facts, then save important findings."
)

# 5. Run with a 'thread_id' to maintain separate conversation sessions
config = {"configurable": {"thread_id": "pro_session_1"}}

def run_research(query):
    print(f"\nUser: {query}")
    # Using invoke() for clean final output
    result = agent.invoke({"messages": [("user", query)]}, config=config)
    print(f"Agent: {result['messages'][-1].content}")

# --- Test Drive ---
# The agent will search for the news, then call save_note automatically
run_research("What is the latest news about Iran vs USA? Save a summary for me.")