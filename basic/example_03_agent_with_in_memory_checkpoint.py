from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_agent # New unified import
from langgraph.checkpoint.memory import MemorySaver # For local session memory

llm = ChatOllama(model="deepseek-v3.1:671b-cloud", base_url="http://localhost:11434")

# 2. Setup Tools
tools = load_tools(["llm-math"], llm=llm)

# 3. Initialize Memory Saver (Checkpointer)
memory = MemorySaver()


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

# Interaction 1: Give information
agent_executor.invoke(
    {"messages": [{"role": "user", "content": "My name is Eason."}]}, 
    config
)

# Interaction 2: Test memory
result = agent_executor.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]}, 
    config
)

print(result["messages"][-1].content)