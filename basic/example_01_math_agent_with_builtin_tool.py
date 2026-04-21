from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_agent # The new standard

llm = ChatOllama(model="deepseek-v3.1:671b-cloud", base_url="http://localhost:11434")
tools = load_tools(["llm-math"], llm=llm)

# create_agent doesn't need AgentType; it uses a system prompt to define behavior
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant that uses tools to answer questions."
)

# Invoke the agent
result = agent.invoke({"messages": [{"role": "user", "content": "What is 15% of 450?"}]})

print(result["messages"][-1].content)