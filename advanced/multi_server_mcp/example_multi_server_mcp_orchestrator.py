
import asyncio
import logging
from langchain_core import messages
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting main function")
    # 1. Connect to multiple specialized MCP servers
    # Server A: Handles Database operations
    # Server B： Handles External Communications(e.g., Slack/Email)

    client = MultiServerMCPClient({
        "db_server": {
            "command": "python",
            "args": ["database_mcp_server.py"], # Hypothetical local server
            "transport": "stdio"
        },
        "notify_server": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-slack"], # Official Slack server
            "transport": "stdio",
            "env": {
                "SLACK_BOT_TOKEN": "place your token here",
                "SLACK_TEAM_ID": "n8n-mac-bot"  # Found in your Slack workspace settings
            }
        }
    })
    logger.info("MCP client initialized")
    # 2. Automatically discover all tools from both servers
    all_tools = await client.get_tools()
    logger.info(f"Discovered {len(all_tools)} tools: {[t.name for t in all_tools]}")
    llm = ChatOllama(model="qwen3.5:397b-cloud", base_url="http://localhost:11434", temperature=0)
    logger.info("LLM initialized")

    # 3. Create specialized worker agents
    # The 'Researcher' gets DB tools; the 'Messenger' gets all non-DB tools
    db_tool_names = {"execute_query"}
    db_tools = [t for t in all_tools if t.name in db_tool_names]
    slack_tools = [t for t in all_tools if t.name not in db_tool_names]
    logger.info(f"DB tools: {len(db_tools)} - {[t.name for t in db_tools]}")
    logger.info(f"Messenger tools: {len(slack_tools)} - {[t.name for t in slack_tools]}")
    if not slack_tools:
        logger.warning("No non-DB tools were detected for the Messenger agent. Check Slack server startup and tool discovery.")

    db_worker = create_agent(llm, 
                             tools=db_tools, 
                             system_prompt="You are a Database Specialist. You have permission to use tools to query the database. Do not say you cannot access it.")
    slack_worker = create_agent(llm, 
                                tools=slack_tools, 
                                system_prompt="""You are a Slack Messenger. 
    You have received user data from the Database Worker. 
    Use any available messaging or notification tools to contact the user. 
    If you don't have a channel ID, use a channel listing tool first.""")
    logger.info("Agents created")

    # 4. Define the supervisor Node
    def superviser_router(state):
        messages = state["messages"]
        last_msg_raw = messages[-1]
        
        # Extract content safely whether it's a tuple ("user", "text") 
        # or a Message Object (HumanMessage, AIMessage)
        if isinstance(last_msg_raw, tuple):
            last_msg_content = last_msg_raw[1]
        else:
            last_msg_content = getattr(last_msg_raw, "content", str(last_msg_raw))
        
        content = last_msg_content.lower()
        logger.info(f"Router examining content: {content[:50]}...")
        
        # If the last message is from the user, go to DB
        if len(messages) == 1:
            return "DB_Worker"
        
        # If DB found data (ID/Email), go to Messenger
        if "id" in content or "email" in content:
            # Prevent infinite loops: only go to Messenger if we haven't tried yet
            if "success" not in content:
                return "Messenger"
                
        return "FINISH"

    # 2. Add a Supervisor Node (The Brain)
    async def supervisor_node(state):
        # This node does nothing but pass the state through
        # so the conditional_edges can run from it
        return state
    
    # 5. Build the complext Graph
    async def db_worker_node(state):
        logger.info("Executing DB_Worker node")
        result = await db_worker.ainvoke(state)
        logger.debug(f"DB_Worker result: {result}")
        return result

    async def slack_worker_node(state):
        logger.info("Executing Messenger node")
        result = await slack_worker.ainvoke(state)
        logger.debug(f"Messenger result: {result}")
        return result

    builder = StateGraph(dict)
    builder.add_node("Supervisor", supervisor_node) # Central Hub
    builder.add_node("DB_Worker", db_worker_node)
    builder.add_node("Messenger", slack_worker_node)

    # START goes to Supervisor
    builder.add_edge(START, "Supervisor")

    # Supervisor decides who to call
    builder.add_conditional_edges(
        "Supervisor",
        superviser_router,
        {
            "DB_Worker": "DB_Worker",
            "Messenger": "Messenger",
            "FINISH": END
        }
    )

    # IMPORTANT: Workers return to Supervisor to re-evaluate
    builder.add_edge("DB_Worker", "Supervisor")
    builder.add_edge("Messenger", "Supervisor")

    # 6. Compile with persistence (optional)
    async with AsyncSqliteSaver.from_conn_string("multi_mcp.db") as memory:
        graph = builder.compile(checkpointer=memory)
        logger.info("Graph compiled")

        # 7. Execute a cross-system task
        config = {"configurable": {"thread_id": "multi_mcp_session_1"}}
        query = "Query the user table for 'Eason' and send him a Slack message if found."
        logger.info(f"Executing query: {query}")

        async for chunk in graph.astream({"messages": [("user", query)]}, config):
            logger.info(f"Graph chunk: {chunk}")
            print(chunk)

if __name__ == "__main__":
    asyncio.run(main())
