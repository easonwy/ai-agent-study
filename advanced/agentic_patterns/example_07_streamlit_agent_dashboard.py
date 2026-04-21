from langchain.agents import create_agent
import streamlit as st
import sqlite3
from langchain_ollama  import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
import logging

# set up debug logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('web_dashboard')


# --- 1. SET UP THE AGENT ENGINE ---
class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]  # Accumulate messages over time


@st.cache_resource
def init_agent():
    llm = ChatOllama(model="qwen3.5:397b-cloud", base_url="http://localhost:11434", temperature=0)
    # Simple specialist for the demo
    researcher = create_agent(model=llm, tools = [])

    builder = StateGraph(AgentState)
    builder.add_node("Researcher", lambda state: {"messages": [researcher.invoke(state)["messages"][-1]]})
    builder.add_edge(START, "Researcher")
    builder.add_edge("Researcher", END)

    memory = SqliteSaver(sqlite3.connect("dashboard_db.db", check_same_thread=False))
    return builder.compile(checkpointer=memory, interrupt_before=["Researcher"])

agent = init_agent()
config = {"configurable": {"thread_id": "web_user_001"}}  # Unique session ID


# --- 2. STREAMLIT UI LAYOUT ---
st.set_page_config(page_title="AI Agent Command Center", layout="wide")
st.title("🤖 Multi-Agent Dashboard")


if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.debug("Initialized session_state.messages (empty)")

# Sidebar for system status & logs
with st.sidebar:
    st.header("System Status")
    st.status("Ollama Connected", state="complete")
    if st.button("Clear History"):
            logger.debug("Clear History pressed - clearing session messages")
            st.session_state.messages = []
            st.rerun()

# Display Chat History
logger.debug(f"Rendering chat history: {len(st.session_state.messages)} messages")
for msg in st.session_state.messages:
    logger.debug(f"Render message role={msg.get('role')} preview={str(msg.get('content'))[:120]}")
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 3. THE CHAT LOGIC ---
if prompt := st.chat_input("What should the team do?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke Agent
    with st.chat_message("assistant"):
        with st.spinner("Team is thinking..."):
            # We use the thread_id to resume or start
            input_data = {"messages": [HumanMessage(content=prompt)]}
            
            # Run the agent (it will stop at the breakpoint if configured)
            logger.debug("Invoking agent.stream for user prompt")
            try:
                final_msg = ""
                for event in agent.stream(input_data, config, stream_mode="values"):
                    logger.debug(f"Stream event: {event}")
                    # event sometimes is {'messages': [...]} or {node: {'messages': [...]} }
                    # handle top-level 'messages' lists
                    if isinstance(event, dict) and "messages" in event and isinstance(event["messages"], list):
                        msgs = event["messages"]
                        if msgs:
                            raw_msg = msgs[-1]
                            if isinstance(raw_msg, HumanMessage):
                                continue
                            content = getattr(raw_msg, "content", str(raw_msg))
                            name = getattr(raw_msg, "name", None)
                            role = "assistant" if isinstance(raw_msg, AIMessage) else getattr(raw_msg, "role", "assistant")

                    # otherwise iterate node states
                    for node_state in event.values() if isinstance(event, dict) else []:
                        # node_state may itself be a list of messages
                        if isinstance(node_state, list):
                            msgs = node_state
                            if msgs:
                                raw_msg = msgs[-1]
                                content = getattr(raw_msg, "content", str(raw_msg))
                                name = getattr(raw_msg, "name", None)
                                role = getattr(raw_msg, "role", "assistant")
                                display = f"[{name}] {content}" if name else content
                                final_msg = display
                                logger.debug(f"Extracted from list node_state: role={role} name={name} preview={display[:120]}")
                                break
                        elif isinstance(node_state, dict) and "messages" in node_state and node_state["messages"]:
                            msgs = node_state["messages"]
                            raw_msg = msgs[-1]
                            # skip human message repeats
                            if isinstance(raw_msg, HumanMessage):
                                continue
                            content = getattr(raw_msg, "content", str(raw_msg))
                            name = getattr(raw_msg, "name", None)
                            role = "assistant" if isinstance(raw_msg, AIMessage) else getattr(raw_msg, "role", "assistant")
                            display = f"[{name}] {content}" if name else content
                            final_msg = display
                            logger.debug(f"Extracted from dict node_state: role={role} name={name} preview={display[:120]}")
                            break

                # Only append if we actually extracted something
                if final_msg:
                    st.markdown(final_msg)
                    st.session_state.messages.append({"role": "assistant", "content": final_msg})
                    logger.debug(f"Appended assistant message after prompt: {str(final_msg)[:200]}")
                else:
                    logger.debug("No assistant message extracted from stream for prompt")
            except Exception as e:
                logger.exception("Error while streaming agent for prompt")
                st.error(f"Agent error: {e}")
    

# --- 4. HUMAN-IN-THE-LOOP PANEL ---
snapshot = agent.get_state(config)
if snapshot.next:
    st.warning("⚠️ The Agent is waiting for your approval to proceed with Research.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Action", use_container_width=True):
            st.info("Task Approved.")
            logger.debug("Approve clicked - resuming agent stream")
            with st.spinner("Resuming agent..."):
                # Consume the agent stream to resume processing until it finishes or hits next breakpoint
                try:
                    for event in agent.stream(None, config, stream_mode="values"):
                        logger.debug(f"Resume stream event: {event}")
                        # event might be {'messages': [...]}
                        if isinstance(event, dict) and "messages" in event and isinstance(event["messages"], list):
                            msgs = event["messages"]
                            if msgs:
                                raw_msg = msgs[-1]
                                # ignore user messages
                                if isinstance(raw_msg, HumanMessage):
                                    continue
                                try:
                                    content = getattr(raw_msg, "content", str(raw_msg))
                                except Exception:
                                    content = str(raw_msg)
                                role = "assistant" if isinstance(raw_msg, AIMessage) else getattr(raw_msg, "role", "assistant")
                                name = getattr(raw_msg, "name", None)
                                display_content = f"[{name}] {content}" if name else content
                                st.session_state.messages.append({"role": role, "content": display_content})
                                logger.debug(f"Appended resumed message (top) role={role} name={name} preview={display_content[:120]}")
                                with st.chat_message(role):
                                    st.markdown(display_content)
                                continue
                        # otherwise iterate over node states which might be lists or dicts
                        for node_state in event.values() if isinstance(event, dict) else []:
                            if isinstance(node_state, list):
                                if node_state:
                                    raw_msg = node_state[-1]
                                    if isinstance(raw_msg, HumanMessage):
                                        continue
                                    try:
                                        content = getattr(raw_msg, "content", str(raw_msg))
                                    except Exception:
                                        content = str(raw_msg)
                                    role = "assistant" if isinstance(raw_msg, AIMessage) else getattr(raw_msg, "role", "assistant")
                                    name = getattr(raw_msg, "name", None)
                                    display_content = f"[{name}] {content}" if name else content
                                    st.session_state.messages.append({"role": role, "content": display_content})
                                    logger.debug(f"Appended resumed message (list) role={role} name={name} preview={display_content[:120]}")
                                    with st.chat_message(role):
                                        st.markdown(display_content)
                                    break
                            elif isinstance(node_state, dict) and "messages" in node_state and node_state["messages"]:
                                msgs = node_state["messages"]
                                raw_msg = msgs[-1]
                                if isinstance(raw_msg, HumanMessage):
                                    continue
                                try:
                                    content = getattr(raw_msg, "content", str(raw_msg))
                                except Exception:
                                    content = str(raw_msg)
                                role = "assistant" if isinstance(raw_msg, AIMessage) else getattr(raw_msg, "role", "assistant")
                                name = getattr(raw_msg, "name", None)
                                display_content = f"[{name}] {content}" if name else content
                                st.session_state.messages.append({"role": role, "content": display_content})
                                logger.debug(f"Appended resumed message (dict) role={role} name={name} preview={display_content[:120]}")
                                with st.chat_message(role):
                                    st.markdown(display_content)
                except Exception as e:
                    logger.exception("Error while resuming agent stream")
                    st.error(f"Error resuming agent: {e}")
            st.rerun()
    with col2:
        if st.button("❌ Terminate", use_container_width=True):
            # Logic to clear state could go here
            st.info("Task cancelled.")