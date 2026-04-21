# Smart Helpdesk Agent

This folder contains a support-agent example that combines:

- LangGraph for multi-step orchestration
- MCP for tool access through a separate support server
- Streamlit for the chat UI
- Ollama-compatible models for triage, response generation, and tone review

## Available Version

- [v1](</Users/easonwu/Dev/personal/ai-agent-study/realworld/smart_helpdesk_agent/v1/README.md>): The current runnable implementation with:
  - triage for intent and sentiment
  - tool-assisted order lookup
  - critic-guided response revision
  - capped retries to avoid infinite loops
  - request-sequence logging across app, graph, and MCP server

## Start Here

If you want to run the example, use the versioned guide:

- [realworld/smart_helpdesk_agent/v1/README.md](/Users/easonwu/Dev/personal/ai-agent-study/realworld/smart_helpdesk_agent/v1/README.md:1)

From the `v1` directory:

```bash
python initialize_support_database.py
streamlit run streamlit_support_app.py
```

## Main Files In `v1`

- [streamlit_support_app.py](/Users/easonwu/Dev/personal/ai-agent-study/realworld/smart_helpdesk_agent/v1/streamlit_support_app.py:1): Streamlit UI and turn orchestration
- [support_workflow.py](/Users/easonwu/Dev/personal/ai-agent-study/realworld/smart_helpdesk_agent/v1/support_workflow.py:1): LangGraph workflow and retry logic
- [support_tools_mcp_server.py](/Users/easonwu/Dev/personal/ai-agent-study/realworld/smart_helpdesk_agent/v1/support_tools_mcp_server.py:1): MCP-backed support tools
- [initialize_support_database.py](/Users/easonwu/Dev/personal/ai-agent-study/realworld/smart_helpdesk_agent/v1/initialize_support_database.py:1): demo database initialization
