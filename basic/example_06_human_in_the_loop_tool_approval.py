import os
import sqlite3
from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver


from langchain_core.tools import tool
from pydantic import BaseModel, Field


file_path = os.path.join(os.path.dirname(__file__), "tasks.txt")

# Define the input schema for the tool using Pydantic
class TaskSchema(BaseModel):
    task_name: str = Field(..., description="The name of the task to save")
    priority: str = Field(..., description="Priority: High, Medium, or Low")

@tool(args_schema=TaskSchema)
def save_task_to_file(task_name: str, priority: str) -> str:
    """Save a task to a local file with its priority."""

    with open(file_path, "a") as f:
        f.write(f"Task: {task_name} | Priority: {priority}\n")
    return f"Task '{task_name}' with priority '{priority}' saved successfully."


@tool
def read_tasks() -> str:
    """Reads all saved tasks from the local file."""
    if not os.path.exists(file_path): return "No tasks found."
    with open(file_path, "r") as f:
        return f.read()


llm = ChatOllama(model="deepseek-v3.1:671b-cloud", base_url="http://localhost:11434")


# 2. Setup SQLite Persistence
# The 'with' block ensures the connection closes properly
conn = sqlite3.connect("memory.db", check_same_thread=False)
memory = SqliteSaver(conn)


# 4. Create the Agent
# In v1.0, create_agent handles the prompt and memory loop internally
agent = create_agent(
    model=llm,
    tools=[save_task_to_file, read_tasks], 
    checkpointer=memory,
    # This PAUSES the agent before it actually writes to the file
    interrupt_before=["tools"],
    system_prompt="You are a helpful assistant. If the user asks to save a task or remind them of something, YOU MUST use the save_task_to_file tool."
)

# 5. Run with a 'thread_id' to maintain separate conversation sessions
config = {"configurable": {"thread_id": "pro_session_1"}}

def run_manager(input_text):
    print(f"\nUser: {input_text}")

    # We use a loop because if interrupted, we need to resume
    for event in agent.stream({"messages": [("user", input_text)]}, config, stream_mode="values"):
        last_msg = event["messages"][-1]

    # Check if we hit a breakpoint (interrupt)
    snapshot = agent.get_state(config)
    if snapshot.next:
        print(f"--- PAUSED: Agent wants to call {snapshot.next} ---")
        choice = input("Approve this action? (y/n): ")
        if choice.lower() == 'y':
            # Resume by passing None to continue the current state
            for event in agent.stream(None, config, stream_mode="values"):
                last_msg = event["messages"][-1]
        else:
            print("Action rejected by user.")
            return

    print(f"Agent: {last_msg.content}")

# Test interactions
run_manager("Add 'Fix the bug' with high priority.")
run_manager("Show me all my tasks.")