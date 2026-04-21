import asyncio
import logging
import uuid
from pathlib import Path

import nest_asyncio
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from support_workflow import SupportEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("AppSupport")

APP_DIR = Path(__file__).resolve().parent
MEMORY_DB_PATH = APP_DIR / "support_mem.db"
MCP_SERVER_PATH = APP_DIR / "support_tools_mcp_server.py"

# Use standard asyncio event loop policy (not uvloop) for Streamlit compatibility
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
# Allow nested event loops for Streamlit compatibility
nest_asyncio.apply()


@st.cache_resource
def get_resources():
    logger.info("[Bootstrap] Creating shared support engine and MCP client.")
    client = MultiServerMCPClient(
        {
            "support": {
                "command": "python",
                "args": [str(MCP_SERVER_PATH)],
                "transport": "stdio",
            }
        }
    )
    return SupportEngine(), client


st.set_page_config(page_title="Smart-Support AI")
st.title("🎧 Smart-Support Helpdesk")

engine, mcp_client = get_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "last_response" not in st.session_state:
    st.session_state.last_response = None


async def chat_logic(prompt: str):
    logger.info(
        "[Turn %s] Received user prompt (%d chars).",
        st.session_state.thread_id,
        len(prompt),
    )
    final_response = None

    async with AsyncSqliteSaver.from_conn_string(str(MEMORY_DB_PATH)) as saver:
        logger.info("[Turn %s] Opening graph with checkpoint store at %s.", st.session_state.thread_id, MEMORY_DB_PATH)
        graph = await engine.get_graph(mcp_client, saver)
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        logger.info("[Turn %s] Starting graph stream.", st.session_state.thread_id)
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=prompt)]},
            config,
            stream_mode="values",
        ):
            messages = chunk.get("messages", [])
            if not messages:
                continue

            msg = messages[-1]
            preview = str(msg.content)[:80].replace("\n", " ")
            logger.info(
                "[Turn %s] Stream event: %s | %s",
                st.session_state.thread_id,
                msg.__class__.__name__,
                preview,
            )

            if isinstance(msg, AIMessage):
                final_response = msg.content

    if not final_response:
        raise RuntimeError("The support workflow completed without producing a final AI response.")

    logger.info("[Turn %s] Final AI response captured.", st.session_state.thread_id)
    st.session_state.last_response = final_response


# UI Loop
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)

if user_input := st.chat_input("How can I help you today?"):
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.last_response = None

    with st.spinner("Agent is working..."):
        try:
            asyncio.run(chat_logic(user_input))
            msg = AIMessage(content=st.session_state.last_response)
            st.session_state.messages.append(msg)
            logger.info("[Turn %s] Response delivered to Streamlit session.", st.session_state.thread_id)
        except Exception as exc:
            logger.exception("[Turn %s] Support workflow failed.", st.session_state.thread_id)
            error_message = (
                "I ran into an internal issue while processing your request. "
                "Please try again in a moment."
            )
            st.session_state.messages.append(AIMessage(content=error_message))
            with st.chat_message("assistant"):
                st.write(error_message)
            st.error(str(exc))
        else:
            st.rerun()
