import sqlite3
import hashlib
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# --- 1. PROMPT REPOSITORY ---
PROMPTS = {
    "A": "You are a concise assistant. Answer in 1-2 sentences only.",
    "B": "You are a detailed mentor. Provide a thorough explanation with a 'Pro Tip' at the end."
}

# --- 2. THE A/B ROUTER LOGIC ---
def get_user_version(thread_id: str) -> str:
    """Consistently assigns a user to group A or B based on their ID."""
    # Use a hash to ensure 'user_1' always gets the same version
    hash_val = int(hashlib.md5(thread_id.encode()).hexdigest(), 16)
    return "A" if hash_val % 2 == 0 else "B"

# --- 3. THE AGENT SETUP ---
llm = ChatOllama(model="qwen3.5:397b-cloud", temperature=0)

def call_agent(state, config):
    thread_id = config["configurable"].get("thread_id", "default")
    version = get_user_version(thread_id)
    
    print(f"--- 🧪 USER {thread_id} ROUTED TO VERSION {version} ---")
    
    messages = [SystemMessage(content=PROMPTS[version])] + state["messages"]
    response = llm.invoke(messages)
    
    # We tag the response so we can track it in our DB
    return {"messages": [response], "current_version": version}

# --- 4. BUILD THE GRAPH ---
class ABState(dict):
    messages: list
    current_version: str

builder = StateGraph(ABState)
builder.add_node("agent", call_agent)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

graph = builder.compile()

# --- 5. SIMULATING A/B TRAFFIC ---
def simulate_traffic():
    # User 1 and User 2 will be routed differently
    users = ["user_eason_01", "user_alice_99"]
    
    for uid in users:
        config = {"configurable": {"thread_id": uid}}
        inputs = {"messages": [HumanMessage(content="Explain what a neural network is.")]}
        
        result = graph.invoke(inputs, config)
        version = result["current_version"]
        content = result["messages"][-1].content
        
        print(f"\n[Result for {uid} (Group {version})]:")
        print(content)
        print("-" * 30)

if __name__ == "__main__":
    simulate_traffic()