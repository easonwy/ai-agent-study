import os
import sqlite3
from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver


from langchain_core.tools import tool
from pydantic import BaseModel, Field


# Define the input schema for the tool using Pydantic
class TaskSchema(BaseModel):
    task_name: str = Field(..., description="The name of the task to save")
    priority: str = Field(..., description="Priority: High, Medium, or Low")

@tool(args_schema=TaskSchema)
def save_task_to_file(task_name: str, priority: str) -> str:
    """Save a task to a local file with its priority."""

    # Ensure file is created in current directory
    file_path = os.path.join(os.path.dirname(__file__), "tasks.txt")

    with open(file_path, "a") as f:
        f.write(f"Task: {task_name} | Priority: {priority}\n")
    return f"Task '{task_name}' with priority '{priority}' saved successfully."


llm = ChatOllama(model="deepseek-v3.1:671b-cloud", base_url="http://localhost:11434")


# 2. Setup SQLite Persistence
# The 'with' block ensures the connection closes properly
conn = sqlite3.connect("memory.db", check_same_thread=False)
memory = SqliteSaver(conn)


# 4. Create the Agent
# In v1.0, create_agent handles the prompt and memory loop internally
agent = create_agent(
    model=llm,
    tools=[save_task_to_file], 
    checkpointer=memory,
    system_prompt="You are a helpful assistant. If the user asks to save a task or remind them of something, YOU MUST use the save_task_to_file tool."
)

# 5. Run with a 'thread_id' to maintain separate conversation sessions
config = {"configurable": {"thread_id": "manager_001"}}

def run_manager(input_text):
    print(f"\nUser: {input_text}")
    inputs = {"messages": [("user", input_text)]}
    
    # 1. Capture the last update from the stream
    final_output = None
    for chunk in agent.stream(inputs, config=config):
        final_output = chunk 

    # 2. Extract the message regardless of which node produced it
    # We look inside the node's dictionary (e.g., final_output["agent"])
    node_name = list(final_output.keys())[0]
    message = final_output[node_name]["messages"][-1]

    print(f"Agent: {message.content}")

# Test interactions
run_manager("Remind me to buy milk. It's high priority.")
run_manager("What was the last task I asked you to save?")