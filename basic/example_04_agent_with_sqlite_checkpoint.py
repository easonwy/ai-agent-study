import sqlite3
from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_agent # New unified import
from langgraph.checkpoint.sqlite import SqliteSaver

llm = ChatOllama(model="deepseek-v3.1:671b-cloud", base_url="http://localhost:11434")

# 2. Setup Tools
tools = load_tools(["llm-math"], llm=llm)

# 2. Setup SQLite Persistence
# The 'with' block ensures the connection closes properly
conn = sqlite3.connect("memory.db", check_same_thread=False)
memory = SqliteSaver(conn)


# 4. Create the Agent
# In v1.0, create_agent handles the prompt and memory loop internally
agent_executor = create_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    system_prompt="You are a helpful assistant. Use tools if needed."
)

# 5. Run with a 'thread_id' to maintain separate conversation sessions
config = {"configurable": {"thread_id": "user_session_1"}}

def chat(user_input):
    print(f"\nUser: {user_input}")
    # We pass the input as a list of messages
    inputs = {"messages": [("user", user_input)]}
    
    for event in agent_executor.stream(inputs, config=config, stream_mode="values"):
        final_msg = event["messages"][-1]
    
    print(f"Agent: {final_msg.content}")

# --- Test Drive ---
# Run this once, then comment out 'My name is Eason' and run again to see it remember!
chat("My name is Eason.")
chat("What is my name?")
chat("What is 15% of 450?")