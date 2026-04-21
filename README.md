# AI Agent Study

This repository is a hands-on study project for building AI agents with LangChain, LangGraph, MCP, Streamlit, local vector stores, and Ollama-compatible models.

It is organized as a progression:

- `basic/`: small focused examples for tools, memory, human approval, supervisors, and simple automation patterns
- `advanced/`: more complex agent architectures such as replanning, self-correcting RAG, MCP orchestration, prompt experiments, and evaluation
- `realworld/`: larger end-to-end demos that feel more like applications than isolated examples

## Project Structure

### `basic/`

Beginner-friendly runnable examples with clearer sequencing and filenames.

- [basic/README.md](/Users/easonwu/Dev/personal/ai-agent-study/basic/README.md:1)
- Representative examples:
  - [example_00_lead_generation_agent.py](/Users/easonwu/Dev/personal/ai-agent-study/basic/example_00_lead_generation_agent.py:1)
  - [example_04_agent_with_sqlite_checkpoint.py](/Users/easonwu/Dev/personal/ai-agent-study/basic/example_04_agent_with_sqlite_checkpoint.py:1)
  - [example_06_human_in_the_loop_tool_approval.py](/Users/easonwu/Dev/personal/ai-agent-study/basic/example_06_human_in_the_loop_tool_approval.py:1)
  - [example_10_dynamic_supervisor_team.py](/Users/easonwu/Dev/personal/ai-agent-study/basic/example_10_dynamic_supervisor_team.py:1)

### `advanced/`

Pattern-oriented examples grouped by topic.

- [advanced/README.md](/Users/easonwu/Dev/personal/ai-agent-study/advanced/README.md:1)
- Main groups:
  - `agentic_patterns/`: supervisor teams, agentic RAG, dynamic tool creation, Streamlit dashboards
  - `self_correcting_rag_workflows/`: retrieval grading, vector DB flows, hybrid local-plus-web RAG
  - `planning_workflows/`: planning, replanning, budget guards, and human approval loops
  - `mcp_basics/`: minimal MCP client/server example
  - `multi_server_mcp/`: multiple MCP servers coordinated by an agent graph
  - `prompt_experiments/`: A/B routing and self-improving prompts
  - `evaluation/`: evaluation and scoring workflows

### `realworld/`

Application-style demos with more realistic workflows and UI.

- [realworld/local_private_financial_analyst/README.md](/Users/easonwu/Dev/personal/ai-agent-study/realworld/local_private_financial_analyst/README.md:1)
- [realworld/smart_helpdesk_agent/README.md](/Users/easonwu/Dev/personal/ai-agent-study/realworld/smart_helpdesk_agent/README.md:1)

Current real-world examples:

- `local_private_financial_analyst/`: private finance analysis over local PDFs and SQLite data, with MCP-backed transaction queries and Streamlit/CLI entrypoints
- `smart_helpdesk_agent/`: a support workflow with triage, tool usage, critique/revision, and a Streamlit chat app

## Setup

### Prerequisites

- Python 3.10 or higher
- An environment where `pip install -r requirements.txt` works
- For most local-model examples: an Ollama-compatible endpoint available at `http://localhost:11434`
- For the lead-generation example in `basic/`: a `GOOGLE_API_KEY` for Gemini

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to run the Gemini-based lead generation sample, add a `.env` file at the project root:

```env
GOOGLE_API_KEY=your_api_key_here
```

## Quick Start

Try one example from each layer:

```bash
python basic/example_01_math_agent_with_builtin_tool.py
python advanced/mcp_basics/example_mcp_weather_client.py
python advanced/planning_workflows/example_dynamic_replanning_agent.py
```

For the lead generation example:

```bash
python basic/example_00_lead_generation_agent.py
```

For the real-world demos, follow the folder-specific READMEs because they have their own setup steps and launch commands.

## Notes

- Many examples are intentionally self-contained and may use their own local SQLite files for checkpoints or demo data.
- Several scripts assume they are run from their own folder so relative paths resolve correctly.
- This repository contains study code and experiments, so some examples are polished teaching demos while others are more exploratory.
