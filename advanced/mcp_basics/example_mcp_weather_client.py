import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_ollama import ChatOllama


async def main():
    # 1. Initialize the MCP Client and point it to your server script
    # This automatically "discovers" any @mcp.too defined in the server
    client = MultiServerMCPClient({
        "weather": {
            "command": "python",
            "args": ["example_mcp_weather_server.py"],
            "transport": "stdio"
        }
    })

    # 2. Fetch the tools from the MCP Server
    tools = await client.get_tools()

    # 3. Setup your local LLM (Ollama)
    llm = ChatOllama(model="qwen3.5:397b-cloud", base_url="http://localhost:11434", temperature=0)

    # 4. Create the agent with the dynamic MCP tools
    agent = create_agent(
        llm,
        tools=tools
    )

    # 5. Run a query
    response = await agent.ainvoke({"messages": [("user", "What's the weather in Singapore?")]})
    print(f"Agent Response: {response['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
