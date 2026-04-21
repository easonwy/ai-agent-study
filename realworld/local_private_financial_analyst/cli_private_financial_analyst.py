import asyncio
from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
# FIX: Using the unified agent creator
from langchain.agents import create_agent 
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# --- 1. STATE DEFINITION ---
class AnalystState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    approved: bool

# --- 2. LOCAL RAG TOOL ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db_path = "./finance_vector_db"
vectorstore = Chroma(persist_directory=vector_db_path, embedding_function=embeddings, collection_name="finance_docs")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def retrieve_financial_docs(query: str) -> str:
    """Use this to find info in invoices, tax transcripts, or policy PDFs."""
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])

# --- 3. MAIN AGENT LOGIC ---
async def main():
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
    instructions = """You are a Private Financial Analyst.
    1. For exact numbers or spending history, use 'query_transactions' (SQL).
    2. For invoices, tax details, or specific document content, use 'retrieve_financial_docs' (RAG).
    3. Combine data from both sources to provide a complete answer.
    4. ALWAYS be precise with currency values.
    
    STRICT RULES:
        1. You have NO knowledge of the user's finances. 
        2. You MUST use 'query_transactions' to see bank data.
        3. You MUST use 'retrieve_financial_docs' to see PDF data.
        4. DO NOT guess or provide examples. Use the tools or say you don't know.
    """
    # If you don't want to force a tool (because sometimes the user just says "Hi"), you must make the "Forbidden Knowledge" rule absolute.
    system_prompt = """You are a Private Financial Analyst.
        - You have ZERO access to the user's data in your internal memory.
        - You are PROHIBITED from answering any question about money, transactions, or documents using your own knowledge.
        - You MUST use a tool for EVERY query involving data.
        - If you do not use a tool, you are failing your mission."""

    # 4. Persistence
    async with AsyncSqliteSaver.from_conn_string("analyst_memory.db") as checkpointer:
        
        # FIX: Using create_agent with 'prompt' instead of 'state_modifier'
        # In modern LangChain, create_agent handles the React logic automatically
        agent = create_agent(
            llm, 
            tools=all_tools,
            system_prompt=instructions,
            checkpointer=checkpointer,
            interrupt_before=["tools"]
        )

        config = {"configurable": {"thread_id": "eason_fin_session_01"}}
        # Generates a unique ID for every run to ensure no "stale" memory bypasses the tools
        # config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        #query = "Compare my Amazon AWS spending from the SQL database with the tax refund amount in my PDF transcript."
        query = "I see an invoice for $2,750 from February. Is that payment reflected in my transaction database yet?"
        
        print(f"\n{'='*50}\nUSER QUERY: {query}\n{'='*50}")

        # STEP 1: Run to breakpoint
        async for chunk in agent.astream({"messages": [("user", query)]}, config, stream_mode="values"):
            last_msg = chunk["messages"][-1]

            # DEBUG: See what the tool actually returned
            if last_msg.type == "tool":
                print(f"🛠️ Tool Result: {last_msg.content[:100]}...")
            else:
                print("DEBUG: Model is NOT using tools. It is just chatting.")

            if last_msg.type == "ai" and last_msg.content:
                print(f"\n🤖 Analyst Thought:\n{last_msg.content}")

        # STEP 2: Approval
        snapshot = await agent.aget_state(config)
        if snapshot.next:
            print("\n🛑 SECURITY CHECKPOINT: Authorization Required")
            tool_calls = snapshot.values["messages"][-1].tool_calls
            for tc in tool_calls:
                print(f"👉 Action: {tc['name']}({tc['args']})")
            
            auth = input("\nAuthorize these tool calls? (y/n): ")
            
            if auth.lower() == 'y':
                async for chunk in agent.astream(None, config, stream_mode="values"):
                    last_msg = chunk["messages"][-1]
                    if last_msg.type == "ai" and last_msg.content:
                        print(f"\n✅ Final Analysis:\n{last_msg.content}")
                    else:
                        print(last_msg)
                        print("DEBUG: Model is NOT using tools. It is just chatting.")
                
                # CRITICAL: Verify if the graph is actually finished
                final_snapshot = await agent.aget_state(config)
                if not final_snapshot.next:
                    print("\n--- Task Completed Successfully ---")
                else:
                    print("\n❌ Task Failed to Complete.")
            else:
                print("\n❌ Task Terminated.")

if __name__ == "__main__":
    asyncio.run(main())
