import os
import importlib
import sqlite3
import json
from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


# --- STEP 1: THE FORGE (Tool Creation) ---
@tool
def create_new_tool(tool_name: str, code: str) -> str:
    """Create a new python tool dynamically. 
    Code must be a complete Python function with the @tool decorator.

    This helper also ensures that the generated file imports the decorator
    so subsequent imports can load the module successfully.
    """
    file_path = "dynamic_tool_library.py"
    header = "from langchain_core.tools import tool\n\n"

    # If file doesn't exist or is empty, create with header
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, "w") as f:
            f.write(header)
    else:
        # ensure header is present somewhere
        with open(file_path, "r") as f:
            content = f.read()
        if "from langchain_core.tools import tool" not in content:
            # prepend the import
            with open(file_path, "w") as f:
                f.write(header + content)

    # Append the new tool to our local library
    with open(file_path, "a") as f:
        f.write(f"\n\n{code}\n")

    return f"Successfully created tool: {tool_name}. It is now available for use."


# --- STEP 2: THE STATE & SETUP ----
class MetaState(TypedDict):
    task: str
    messages: Annotated[list, lambda x, y: x + y]  # Accumulate messages over time
    available_tools: list[str]


llm = ChatOllama(model="qwen3.5:397b-cloud", base_url="http://localhost:11434", temperature=0)


# --- STEP 3: DYNAMIC LOADING LOGIC ---
def get_current_tools():
    """Dynamically imports and returns all tools from dynamic_tool_library.py"""
    tools = [create_new_tool]
    if os.path.exists("dynamic_tool_library.py"):
        try: 
            # Refresh the module to see new tools
            spec = importlib.util.spec_from_file_location("dynamic_tool_library", "dynamic_tool_library.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find all functions decorated with @tool
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                # include objects produced by the @tool decorator; they expose .name and .invoke
                if hasattr(obj, "name") and hasattr(obj, "invoke") and obj.name != "create_new_tool":
                    tools.append(obj)
        except Exception as e:
            print(f"Error loading custom tools: {e}")
    return tools


# --- STEP 4: NODES ---
def agent_node(state: MetaState):
    print("---- AGENT IS REASONING ----")
    current_tools = get_current_tools()
    tool_names = [t.name for t in current_tools]
    print(f"Available tools: {tool_names}")

    # We use bind_tools to give the LLM access to its current library
    llm_with_tools = llm.bind_tools(current_tools)

    # Add a system prompt explaining the "Forge" capability
    system_prompt = (
        f"Goal: {state['task']}\n"
        f"Available tools: {tool_names}\n"
        "Instructions:\n"
        "1. First, check if you have a tool that can solve this task.\n"
        "2. If NO suitable tool exists, use 'create_new_tool' with:\n"
        "   - tool_name: descriptive name\n"
        "   - code: complete Python function with @tool decorator\n"
        "3. After tool creation, the tool will be available for use.\n"
        "4. Do NOT try to call tools that don't exist."
    )

    try:
        response = llm_with_tools.invoke([HumanMessage(content=system_prompt)] + state.get("messages", []))
        return {"messages": [response]}
    except Exception as e:
        print(f"Agent error: {str(e)}")
        # Return error message and try again
        error_msg = HumanMessage(content=f"Error during reasoning: {str(e)}. Please try a simpler approach.")
        return {"messages": [error_msg]}


def tool_executor_node(state: MetaState):
    """Execute any tool calls the agent decided to use."""
    print("---- EXECUTING TOOL ----")
    last_message = state["messages"][-1]
    
    tool_results = []
    current_tools = get_current_tools()
    tools_dict = {tool.name: tool for tool in current_tools}
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call.get("args", {})
            
            print(f"Executing tool: {tool_name} with input: {tool_input}")
            if tool_name in tools_dict:
                # convert numeric-looking strings to numbers for convenience
                for k, v in list(tool_input.items()):
                    if isinstance(v, str):
                        try:
                            if "." in v or "e" in v.lower():
                                tool_input[k] = float(v)
                            else:
                                tool_input[k] = int(v)
                        except ValueError:
                            pass
                try:
                    result = tools_dict[tool_name].invoke(tool_input)
                    print(f"Tool result: {result}")
                    tool_results.append(ToolMessage(tool_call_id=tool_call["id"], content=str(result)))
                except Exception as e:
                    error_msg = f"Error executing {tool_name}: {str(e)}"
                    print(error_msg)
                    tool_results.append(ToolMessage(tool_call_id=tool_call["id"], content=error_msg))
            else:
                error_msg = f"Tool '{tool_name}' not found. Available tools: {list(tools_dict.keys())}"
                print(error_msg)
                tool_results.append(ToolMessage(tool_call_id=tool_call["id"], content=error_msg))
    
    return {"messages": tool_results}


def should_continue(state: MetaState):
    """Route based on whether agent wants to use a tool or is done."""
    last_message = state["messages"][-1]
    
    # If the last message has tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_executor"
    # Otherwise, we're done
    return "end"


# --- STEP 5: GRAPH BUILDER ---
builder = StateGraph(MetaState)
builder.add_node("Agent", agent_node)
builder.add_node("ToolExecutor", tool_executor_node)

# Starting point
builder.add_edge(START, "Agent")

# Agent decides whether to use a tool or finish
builder.add_conditional_edges(
    "Agent",
    should_continue,
    {
        "tool_executor": "ToolExecutor",
        "end": END
    }
)

# After executing tools, loop back to agent for next reasoning step
builder.add_edge("ToolExecutor", "Agent")


memory = SqliteSaver(sqlite3.connect("meta_agent.db", check_same_thread=False))
graph = builder.compile(checkpointer=memory)


# --- STEP 6: TEST DRIVE ---
config = {"configurable": {"thread_id": "meta_001"}}

def run_meta(task, max_iterations=10):
    print(f"\n User Task: {task}")
    print("=" * 60)
    
    # Reset thread for fresh start
    config["configurable"]["thread_id"] = f"meta_{hash(task) % 10000}"
    
    # Run the graph and stream all events
    iteration = 0
    try:
        for event in graph.stream({"task": task, "messages": [], "available_tools": []}, config):
            iteration += 1
            if iteration > max_iterations:
                print(f"\n⚠️  Max iterations ({max_iterations}) reached.")
                break
            print(f"\n[Iteration {iteration}]")
            for node_name, node_state in event.items():
                print(f"  Node: {node_name}")
                if "messages" in node_state:
                    last_msg = node_state["messages"][-1] if node_state["messages"] else None
                    if last_msg:
                        if hasattr(last_msg, "content"):
                            content_str = str(last_msg.content)
                            print(f"    Content: {content_str[:100]}..." if len(content_str) > 100 else f"    Content: {content_str}")
                        if hasattr(last_msg, "tool_calls"):
                            print(f"    Tool calls: {[tc['name'] for tc in last_msg.tool_calls]}")
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Get final state
    try:
        final_state = graph.get_state(config)
        print("\n" + "=" * 60)
        print("---- FINAL RESULT ----")
        last_msg = final_state.values["messages"][-1] if final_state.values["messages"] else None
        if last_msg and hasattr(last_msg, "content"):
            print(f"Final Answer: {last_msg.content}")
        else:
            print("No final result generated.")
    except Exception as e:
        print(f"Error retrieving final state: {e}")


# Challenge: The agent has no math tools, It must CREATE one.
run_meta("Calculate the volume of a sphere with radius 5.5. Use a custom tool.")


