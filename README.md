# AI Lead Generation Agent

This project uses LangChain and LangGraph to create an AI agent that can search the web for potential leads and save them to a file.

## Prerequisites

- Python 3.10 or higher
- A Google Gemini API Key (get it from [Google AI Studio](https://aistudio.google.com/))

## Installation

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file in the project root and add your API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

Run the main script:
```bash
python basic/example_00_lead_generation_agent.py
```

## How it Works

- **basic/example_00_lead_generation_agent.py**: The entry point. It sets up the `LangGraph` agent which orchestrates the process.
- **basic/lead_generation_tools.py**: Contains the custom tools (`search_tool`, `scrape_tool`, `save_tool`) that the agent uses.
- **Agent Logic**: The agent receives a goal (find leads), breaks it down into steps (search, scrape, analyze), and executes tools until the goal is met. Finally, it uses the `save_tool` to write the results to `leads_output.txt`.
