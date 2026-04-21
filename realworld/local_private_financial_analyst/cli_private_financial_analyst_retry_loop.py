import asyncio
from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain.agents import create_agent 
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# --- 1. STATE DEFINITION ---
class AnalystState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    approved: bool

# --- Part 1: Local RAG (PDFs) ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db_path = "./finance_vector_db"
vectorstore = Chroma(persist_directory=vector_db_path, embedding_function=embeddings, collection_name="finance_docs")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def retrieve_financial_docs(query: str) -> str:
    """Finds info in invoices, tax transcripts, or policy PDFs."""
    docs = retriever.invoke(query)
    content = "\n\n".join([d.page_content for d in docs])
    return content if content.strip() else "No relevant information found in PDF documents."

# --- 3. MAIN AGENT LOGIC ---
async def main_agent_logic():
    # MCP Client Setup
    client = MultiServerMCPClient({
        "db_server": {
            "command": "python",
            "args": ["finance_transactions_mcp_server.py"],
            "transport": "stdio",
        }
    })

    # 1. Discovery
    mcp_tools = await client.get_tools()
    all_tools = mcp_tools + [retrieve_financial_docs]
    
    # 2. Setup LLM
    llm = ChatOllama(model="qwen3.5:397b-cloud", base_url="http://localhost:11434", temperature=0)

    # 3. Instructions
    system_instructions = """You are a Private Financial Analyst.
    - If you need exact numbers/history, use 'query_transactions'.
    - If the DB returns 'No records', you MUST use 'retrieve_financial_docs' to check PDFs.
    - NEVER stop until you have checked BOTH sources if the first one fails."""

    # 4. Persistence
    async with AsyncSqliteSaver.from_conn_string("analyst_memory_loop.db") as checkpointer:
        
        # FIX: Using create_agent with 'prompt' instead of 'state_modifier'
        # In modern LangChain, create_agent handles the React logic automatically
        agent = create_agent(
            llm,
            tools=all_tools,
            system_prompt=system_instructions,
            checkpointer=checkpointer,
            interrupt_before=["tools"]
        )

        config = {"configurable": {"thread_id": "eason_fin_loop_session_01"}}
        # Generates a unique ID for every run to ensure no "stale" memory bypasses the tools
        # config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        query = "Find my AWS spending and compare it with my tax refund in the PDFs."
        # query = "I see an invoice for $2,750 from February. Is that payment reflected in my transaction database yet?"
        
        print(f"\n{'='*50}\nUSER QUERY: {query}\n{'='*50}")

        # --- EXECUTION LOOP ---
        inputs = {"messages": [HumanMessage(content=query)]}

        while True:
            # Run until next interrupt or end
            async for chunk in agent.astream(inputs, config, stream_mode="values"):
                last_msg = chunk["messages"][-1]
                if last_msg.type == "ai" and last_msg.content:
                    print(f"\n🤖 Analyst: {last_msg.content}")
            
            # Check if we hit a Security Breakpoint
            snapshot = await agent.aget_state(config)
            if not snapshot.next:
                break # Graph finished

            # Human-in-the-Loop Approval
            print("\n" + "!"*30 + " SECURITY CHECK " + "!"*30)
            t_calls = snapshot.values["messages"][-1].tool_calls
            print(f"Agent wants to call: {[tc['name'] for tc in t_calls]}")
            
            auth = input("Authorize access to private data? (y/n): ")
            if auth.lower() == 'y':
                inputs = None # Resume from current state
            else:
                print("Access Denied. Terminating.")
                break
        
        print("\n" + "="*50 + "\nTASK COMPLETED\n" + "="*50)

if __name__ == "__main__":
    try:
        asyncio.run(main_agent_logic())
    except KeyboardInterrupt:
        pass
