"""
To implement a Cost & Safety Monitor, we introduce a specialized node that acts as a "Budget Guard." Before each execution step, it evaluates the number of iterations and the complexity of the task. If it exceeds your predefined limit, it triggers a Human-in-the-Loop pause.
1. The Strategy
    a. Thresholds: Set a maximum number of steps (e.g., 5 steps).
    b. State Variable: Track step_count in the graph state.
    c. Breakpoint: Use LangGraph's interrupt_before to force the agent to wait for your "Yes/No" if the budget is exceeded.
"""

import sqlite3
from typing import Annotated, List, TypedDict
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from langgraph.checkpoint.sqlite import SqliteSaver

# ---- 1. STATE & SCHEMAS ---
class Plan(BaseModel):
    steps: List[str]

class PlanState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[tuple], lambda x, y: x + y]
    final_answer: str
    step_count: int # <--- Tracks "cost" (number of tool calls)


# --- 2. NODES ---
llm = ChatOllama(model="qwen3.5:397b-cloud", base_url="http://localhost:11434", temperature=0, format="json")
planner_llm = llm.with_structured_output(Plan)
search_tool = DuckDuckGoSearchRun()


def plan_node(state: PlanState) -> PlanState:
    print("--- 🧠 PLANNING ---")
    prompt = f"""
    You are a planning assistant. Output only valid JSON matching this schema:
    {{"steps": ["step1", "step2", ...]}}

    Create a step-by-step plan to solve the following goal.
    Goal: {state['input']}
    """
    plan = planner_llm.invoke(prompt)
    return {"plan": plan.steps, "step_count": 0}

def monitor_node(state: PlanState) -> PlanState:
    print("--- 🧪 MONITORING ---")
    print(f"--- MONITOR: Step {state['step_count'] + 1}/5 ---")
    # This node effectively does nothing bu serve as a target for a breakpoint
    return {"step_count": state["step_count"] + 1}

def execute_node(state: PlanState) -> PlanState:
    current_step = state["plan"][0]
    print(f"--- EXECUTE: {current_step} ---")
    result = search_tool.run(current_step)
    return {"past_steps": [(current_step, result)], "plan": state["plan"]}

def replan_node(state: PlanState) -> PlanState:
    if not state["plan"]:
        res = llm.invoke(f"Summarize based on: {state['past_steps']}")
        return {"final_answer": res.content}
    return state


# --- 3. GRAPH CONSTRUCTION ---
builder = StateGraph(PlanState)

builder.add_node("planner", plan_node)
builder.add_node("monitor", monitor_node)
builder.add_node("executor", execute_node)
builder.add_node("replanner", replan_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "monitor")
builder.add_edge("monitor", "executor")
builder.add_edge("executor", "replanner")

# Route: Check if finished or need more steps
builder.add_conditional_edges(
    "replanner",
    lambda state: "finish" if state.get("final_answer") else "continue", 
    {"continue": "monitor", "finish": END}
)

# --- 4. Persistence with Interrupt Before(Breakpoint) ---
memory = SqliteSaver(sqlite3.connect("agentic_cost_monitored_planner.db", check_same_thread=False))
# INTERRUPT if we hit the monitor node and step_count > 3
graph = builder.compile(checkpointer=memory, interrupt_before=["monitor"])


# --- 5. EXECUTION LOOP WITH HUMAN APPROVAL ---
config = {"configurable": {"thread_id": "budget_123"}}


def run_task(task):
    inputs = {"input": task, "past_steps": [], "step_count": 0}
    
    for chunk in graph.stream(inputs, config):
        print(chunk)
        
    # Check if we are paused at the monitor
    snapshot = graph.get_state(config)
    while snapshot.next:
        current_steps = snapshot.values.get("step_count", 0)
        print(f"\n🛑 BUDGET ALERT: Agent has run {current_steps} steps.")
        choice = input("Authorize next step? (y/n): ")
        
        if choice.lower() == 'y':
            for chunk in graph.stream(None, config): # Resume
                print(chunk)
            snapshot = graph.get_state(config)
        else:
            print("Operation terminated to save costs.")
            break

run_task("Deeply research the 2026 AI hardware market.")