# Basic Examples

This folder contains small, progressively more advanced agent examples.

## Example Index

- `example_00_lead_generation_agent.py`: a lead-generation agent that uses search, scraping, and file output tools.
- `example_01_math_agent_with_builtin_tool.py`: a minimal agent that uses a built-in math tool.
- `example_02_agent_with_custom_function_tool.py`: an agent that calls a custom Python function as a tool.
- `example_03_agent_with_in_memory_checkpoint.py`: conversation memory using an in-memory checkpointer.
- `example_04_agent_with_sqlite_checkpoint.py`: conversation memory persisted with SQLite.
- `example_05_agent_with_file_save_tool.py`: a custom file-writing tool for saving tasks.
- `example_06_human_in_the_loop_tool_approval.py`: tool execution with user approval before side effects.
- `example_07_research_agent_with_notes_tool.py`: a research agent that searches the web and saves notes.
- `example_08_research_agent_second_query.py`: a second research example using the same pattern with a different prompt.
- `example_09_basic_supervisor_graph.py`: a simple supervisor graph that routes work between agents.
- `example_10_dynamic_supervisor_team.py`: a more dynamic multi-agent supervisor workflow.
- `example_11_code_writer_executor_loop.py`: a writer-executor loop that generates and runs Python code.
- `example_12_screen_automation_vision.py`: a vision-driven screen automation prototype.

## Related Files

- `lead_generation_tools.py`: shared helper tools used by `example_00_lead_generation_agent.py`.
- `inspect_sqlite_checkpoints.py`: SQLite helper utility for inspecting checkpoint data.
