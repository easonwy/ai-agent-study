import os
import uuid

import asyncio
import nest_asyncio
import streamlit as st
from financial_analyst_engine import FinancialAnalystEngine
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Use standard asyncio event loop policy (not uvloop) for Streamlit compatibility
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
# Allow nested event loops for Streamlit compatibility
nest_asyncio.apply()

# --- 1. SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"web_{uuid.uuid4()}"
if "last_ai_content" not in st.session_state:
    st.session_state.last_ai_content = ""
if "next_action" not in st.session_state:
    st.session_state.next_action = None
if "requested_tools" not in st.session_state:
    st.session_state.requested_tools = []

# --- 2. CACHE GLOBAL RESOURCES ---
def get_global_resources():
    engine = FinancialAnalystEngine()
    return engine

# --- 3. ASYNC AGENT EXECUTION ---
async def run_agent_logic(user_input=None, resume=False, max_steps=1, status_container=None):
    print(f"[DEBUG] run_agent_logic called with user_input={user_input}, resume={resume}, max_steps={max_steps}")
    engine = get_global_resources()
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient({
        "db_server": {
            "command": "python",
            "args": ["finance_transactions_mcp_server.py"],
            "transport": "stdio",
        }
    })
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    async with AsyncSqliteSaver.from_conn_string("analyst_memory_app.db") as checkpointer:
        # Get the compiled agent graph
        print("[DEBUG] Getting agent from engine...")
        agent, mcp_client = await engine.get_agent(checkpointer, client)
        print("[DEBUG] Agent retrieved successfully")
        
        # Determine if we are starting a new prompt or resuming from a breakpoint
        inputs = {"messages": [HumanMessage(content=user_input)]} if not resume else None
        
        last_tools = []
        same_tool_count = 0
        total_requested_tools = []
        
        for iteration in range(max_steps):
            print(f"[DEBUG] Starting agent stream iteration {iteration+1}/{max_steps} with inputs={inputs is not None}")
            
            chunk_count = 0
            all_messages = []
            try:
                async for chunk in agent.astream(inputs, config, stream_mode="values"):
                    chunk_count += 1
                    if "messages" in chunk:
                        all_messages = chunk["messages"]
                        last_msg = chunk["messages"][-1]
                        
                        if status_container:
                            if hasattr(last_msg, "type"):
                                if last_msg.type == "ai" and hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                    tools_str = ", ".join([tc['name'] for tc in last_msg.tool_calls])
                                    status_container.write(f"🧠 Agent requested tools: `{tools_str}`")
                                elif last_msg.type == "tool":
                                    preview = str(last_msg.content)[:100].replace('\n', ' ')
                                    status_container.write(f"🛠️ Tool `{last_msg.name}` returned: {preview}...")
                        
                        if last_msg.type == "ai" and last_msg.content:
                            st.session_state.last_ai_content = last_msg.content
            except Exception as e:
                print(f"[DEBUG] Exception during stream: {e}")
            
            print(f"[DEBUG] Stream completed for iteration {iteration+1}. Total chunks: {chunk_count}")
            
            # Check if the graph hit a breakpoint (interrupt)
            snapshot = await agent.aget_state(config)
            st.session_state.next_action = snapshot.next
            print(f"[DEBUG] next_action = {st.session_state.next_action}")
            if snapshot.next:
                st.session_state.requested_tools = []
                if "messages" in snapshot.values:
                    for msg in reversed(snapshot.values["messages"]):
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            st.session_state.requested_tools = [
                                tc['name'] for tc in msg.tool_calls
                            ]
                            break
                total_requested_tools.extend(st.session_state.requested_tools)
            else:
                st.session_state.requested_tools = []
                
            # If no breakpoint was hit or final response generated, stop looping
            if not st.session_state.next_action or st.session_state.last_ai_content:
                break
                
            # Check if we're stuck in a loop (same tools requested multiple times)
            current_tools = set(st.session_state.requested_tools)
            if current_tools == set(last_tools) and current_tools:
                same_tool_count += 1
                if same_tool_count >= 10:
                    print(f"[DEBUG] Stuck in tool loop after {iteration+1} iterations, breaking")
                    break
            else:
                same_tool_count = 0
                last_tools = st.session_state.requested_tools
                
            # Provide None as input for resumption
            inputs = None
            
        # Optional: gracefully close the specific TCP or Subprocess links from client if needed here.
        if hasattr(client, "close") and callable(client.close):
            try:
                await client.close()
            except Exception:
                pass
                
        return total_requested_tools

# --- 5. STREAMLIT UI LAYOUT ---
st.set_page_config(page_title="Private Finance Analyst", layout="wide", page_icon="💸")
st.title("💸 Private Financial Analyst")

# Sidebar: Document Watchdog
with st.sidebar:
    st.header("📂 Document Inbox")
    uploaded = st.file_uploader("Upload Financial PDFs", accept_multiple_files=True)
    if uploaded:
        if not os.path.exists("./private_finance_docs"):
                os.makedirs("./private_finance_docs")
        for file in uploaded:
            with open(f"./private_finance_docs/{file.name}", "wb") as f:
                f.write(file.getbuffer())
        if st.button("🚀 Re-Index Documents", use_container_width=True):
            with st.spinner("Indexing PDFs into Vector DB..."):
                os.system("python ingest_financial_documents.py")
            st.success("Indexing Complete!")
    
    st.divider()
    st.caption(f"Session ID: {st.session_state.thread_id}")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.write(msg.content)

# Handle Approval UI (Security Gate)
if st.session_state.next_action:
    with st.chat_message("assistant"):
        st.warning(f"**Security Checkpoint:** The analyst wants to access your data using: `{st.session_state.requested_tools}`")
        c1, c2 = st.columns(2)
        if c1.button("✅ Authorize Access", use_container_width=True):
            st.session_state.last_ai_content = ""
            with st.status("Execution Process...", expanded=True) as st_status:
                tool_results = list(st.session_state.requested_tools) if st.session_state.requested_tools else []
                async_tools = asyncio.run(run_agent_logic(resume=True, max_steps=30, status_container=st_status))
                if async_tools:
                    tool_results.extend(async_tools)
                
                # Append result and clear gate
                if st.session_state.last_ai_content:
                    st.session_state.messages.append(AIMessage(content=st.session_state.last_ai_content))
                else:
                    # If no final summary, synthesize one from execution
                    summary = f"Analysis completed. Used tools: {', '.join(set(tool_results)) if tool_results else 'none'}."
                    st.session_state.messages.append(AIMessage(content=summary))
                    print(f"[DEBUG] Using synthesized summary: {summary}")
                
                st.session_state.next_action = None
                st.rerun()
        if c2.button("❌ Deny Access", use_container_width=True):
            st.session_state.next_action = None
            st.error("Access denied. Task terminated.")
            st.rerun()

# Handle New User Input
if prompt := st.chat_input("Ask about your finances (e.g., 'What was my total spending last month?')"):
    print(f"[DEBUG] User input received: {prompt}")
    st.session_state.last_ai_content = ""
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.status("Thinking Process...", expanded=True) as st_status:
            print("[DEBUG] Starting agent execution...")
            asyncio.run(run_agent_logic(user_input=prompt, status_container=st_status))
            print(f"[DEBUG] Agent execution completed")
            print(f"[DEBUG] st.session_state.last_ai_content = '{st.session_state.last_ai_content}'")
            print(f"[DEBUG] st.session_state.next_action = {st.session_state.next_action}")
            
            # If no breakpoint was hit, show the final response
            if not st.session_state.next_action:
                st_status.update(label="Process Complete", state="complete", expanded=False)
                print("[DEBUG] No breakpoint, displaying response")
                if st.session_state.last_ai_content:
                    st.write(st.session_state.last_ai_content)
                    st.session_state.messages.append(AIMessage(content=st.session_state.last_ai_content))
                else:
                    st.warning("No response generated. This may indicate an issue with the agent execution.")
            else:
                st_status.update(label="Waiting for Authorization", state="complete", expanded=False)
                print("[DEBUG] Breakpoint hit, rerunning to show approval UI")
                st.rerun() # Refresh to show the Approval UI
