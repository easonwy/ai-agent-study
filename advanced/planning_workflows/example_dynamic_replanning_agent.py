


import sqlite3
from typing import Annotated, List, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field



# ---- 1. SCHEMAS ----
class Plan(BaseModel):
    """The current execution plan."""
    steps: List[str] = Field(description="List of steps to execute in order.")

# ---- 2. STATE ----
class PlanState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[tuple], lambda x, y: x + y]
    steps_executed: int
    max_steps: int
    final_answer: str

# ---- 3. NODES ----
llm = ChatOllama(model="qwen3.5:397b-cloud", base_url="http://localhost:11434", temperature=0, format="json")
planner_llm = llm.with_structured_output(Plan)
search_tool = DuckDuckGoSearchRun()

def plan_node(state: PlanState) -> PlanState:
    print("--- CREATING INITIAL PLAN ---")
    prompt = f"""
    You are a planning assistant. Output only valid JSON matching this schema:
    {{"steps": ["step1", "step2", ...]}}

    Create a step-by-step plan to solve the following goal.
    Goal: {state['input']}
    """
    plan = planner_llm.invoke(prompt)
    return {
        "plan": plan.steps,
        "steps_executed": 0,
        "max_steps": state.get("max_steps", 5),
    }

def execute_node(state: PlanState) -> PlanState:
    current_step = state["plan"][0]
    print(f"--- EXECUTING STEP: {current_step} ----")
    # For demo, we only handle search steps. In production, you'd have more complex logic here.
    result = search_tool.run(current_step)
    return {
        "past_steps": [(current_step, result)],
        "steps_executed": state.get("steps_executed", 0) + 1,
    }

def replan_node(state: PlanState) -> PlanState:
    print(" --- RE-PLANNING & EVALUATING --- ")

    # If no more steps or we reached the maximum step budget, summarize and finish
    if len(state["plan"]) <= 1 or state.get("steps_executed", 0) >= state.get("max_steps", 5):
        if state.get("steps_executed", 0) >= state.get("max_steps", 5) and len(state["plan"]) > 1:
            print(f"--- MAX STEP LIMIT REACHED ({state['steps_executed']}/{state['max_steps']}) ---")
        else:
            print("--- GOAL REACHED ----")
        prompt = f"Based on these results: {state['past_steps']}, provide a final answer to: {state['input']}"
        response = llm.invoke(prompt)
        return {"final_answer": response.content}
    
    # Otherwise, update the plan
    prompt = f"""
    You are a planning assistant. Output only valid JSON matching this schema:
    {{"steps": ["step1", "step2", ...]}}

    Original Goal: {state['input']}
    Completed: {state['past_steps']}
    Remaining Plan: {state['plan'][1:]}

    Update the plan. Add new steps if the results suggest missing information, or remove steps if they are no longer needed.
    """

    new_plan = planner_llm.invoke(prompt)
    return {"plan": new_plan.steps}

# --- 4. GRAPH CONSTRUCTION ---
builder = StateGraph(PlanState)

builder.add_node("planner", plan_node)
builder.add_node("executor", execute_node)
builder.add_node("replanner", replan_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", "replanner")

# Conditional: Continue to next step or Finish
builder.add_conditional_edges(
    "replanner",
    lambda state: "finish" if state.get("final_answer") else "continue", 
    {"continue": "executor", "finish": END}
)


memory = SqliteSaver(sqlite3.connect("planning_agent.db", check_same_thread=False))
graph = builder.compile(checkpointer=memory)


# --- 5. EXECUTING ---
def run_planner(task):
    config = {"configurable": {"thread_id": "plan_01"}}
    inputs = {
        "input": task,
        "plan": [],
        "past_steps": [],
        "steps_executed": 0,
        "max_steps": 5,
        "final_answer": "",
    }
    # If using a stream, make sure 'inputs' is the first arg, and 'config' is the second
    for chunk in graph.stream(inputs, config=config): 
        print(chunk)

run_planner("Investigate the current state of Open Source AI in 2026. Who are the leaders and what is the top model?")
