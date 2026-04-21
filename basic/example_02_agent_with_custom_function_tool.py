from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_agent # The new standard

llm = ChatOllama(model="deepseek-v3.1:671b-cloud", base_url="http://localhost:11434")

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# create_agent doesn't need AgentType; it uses a system prompt to define behavior
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant that uses tools to answer questions."
)

# Invoke the agent
result = agent.invoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]})

print(result["messages"][-1].content)