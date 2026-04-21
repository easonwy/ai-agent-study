import sqlite3
from typing import TypedDict
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langchain_experimental.utilities import PythonREPL


# 1. Setup LLM & Specialists
llm = ChatOllama(model="deepseek-v3.1:671b-cloud", base_url="http://localhost:11434", temperature = 0)
repl = PythonREPL()


@tool
def execute_python(code: str) -> str:
    """Executes Python code and returns the output."""
    try:
        cleaned_code = code.replace("```python", "").replace("```", "").strip()
        result = repl.run(cleaned_code)
        return f"Output: {result}" if result else "Executed successfully with no output."
    except Exception as e:
        return f"Error executing code: {e}"
    

# 2. Define State
class CodeState(TypedDict):
    task: str
    code: str
    execution_result: str
    iterations: int

# 3. Define Nodes
def writer_node(state: CodeState):
    print(f"--- WRITER (Attempt {state['iterations'] + 1}) ---")
    prompt = f"Task: {state['task']}\n\n"
    if state['execution_result']:
        prompt += f"Previous Code: {state['code']}\nError: {state['execution_result']}\nFix the error."
    
    # We want ONLY the raw code
    response = llm.invoke(f"{prompt}\nReturn ONLY the python code block.")
    return {"code": response.content, "iterations": state['iterations'] + 1}

def executor_node(state: CodeState):
    print(f"--- EXECUTOR (Attempt {state['iterations']}) ---")
    # `execute_python` is a StructuredTool created with @tool; call via `.invoke()`
    result = execute_python.invoke(state['code'])
    return {"execution_result": result}

def gatekeeper(state: CodeState):
    print(f"--- GATEKEEPER (Attempt {state['iterations']}) ---")
    # If the output starts with 'Error' and we haven't tried 3 times, go back
    if "Error" in state["execution_result"] and state["iterations"] < 3:
        print(f"Bugs found: {state['execution_result']}. Sending back to Writer...")
        return "REWRITE"
    return "FINISH"

# 4. Build the Graph
builder = StateGraph(CodeState)
builder.add_node("Writer", writer_node)
builder.add_node("Executor", executor_node)

builder.add_edge(START, "Writer")
builder.add_edge("Writer", "Executor")
builder.add_conditional_edges("Executor", gatekeeper, {"REWRITE": "Writer", "FINISH": END})


# 5. Run
memory = SqliteSaver(sqlite3.connect("coder_repl.db", check_same_thread=False))
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "repl_001"}}
initial_state = {"task": "Write a function to divide 10 by 0 and handle the error.", "iterations": 0, "execution_result": ""}

for chunk in graph.stream(initial_state, config):
    print(chunk)


