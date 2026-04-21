"""
Autonomous Market Analyst

The most complex pattern in modern agent development is the Plan-and-Execute (with Re-Planning) pattern. Unlike a simple supervisor, this system creates a multi-step checklist, executes one step, observes the result, and then rewrites the remaining plan if the environment changed or an error occurred.

Planner: Breaks a complex query into 5 sub-tasks.
Executor: Runs a sub-task (e.g., "Find Nvidia's Q4 revenue").
Re-Planner: Looks at the revenue and says, "Wait, they mentioned a new AI chip; I need to add a step to research that specific chip before summarizing."
"""

import sqlite3
from typing import Annotated, List, TypedDict

from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver


# --- STEP 1: SCHEMAS ---
class Plan(BaseModel):
    steps: List[str] = Field(description="The steps to follow, in order.")

class PlanState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[tuple], lambda x, y: x + [y]]
    response: str

# --- STEP 2: LLM SETUP & NODES ---
llm = ChatOllama(model="qwen3.5:397b-cloud", base_url="http://localhost:11434", temperature=0)
planner_llm = llm.with_structured_output(Plan)
search_tool = DuckDuckGoSearchRun()


def plan_node(state: PlanState):
    print("---- PLANNER IS CREATING A PLAN ----")
    prompt = f"Create a step-by-step plan to answer: {state['input']}"
    plan = planner_llm.invoke(prompt)
    return {"plan": plan.steps}


def execute_node(state: PlanState):
    print(f"--- 🚀 EXECUTING: {state['plan'][0]} ---")
    task = state['plan'][0]
    result = search_tool.run(task)
    return {"past_steps": [(task, result)]}


def replan_node(state: PlanState):
    print("--- 🔄 RE-PLANNER IS CHECKING THE PLAN ---")
    if len(state["plan"]) <= 1:
        return {"response": state["past_steps"][-1][1]}
    
    # Otherwise, update the plan based on what we just learned
    prompt = f"""
    Original Goal: {state['input']}
    Current Plan: {state['plan']}
    Completed: {state['past_steps']}
    Update the plan. Remove the first step. If the results suggest new info is needed, add steps.
    """
    new_plan = planner_llm.invoke(prompt)
    return {"plan": new_plan.steps}

def should_continue(state: PlanState):
    if state.get("response"):
        return "FINISH"
    return "CONTINUE"

# --- STEP 3: BUILD THE COMPLEX GRAPH ---
builder = StateGraph(PlanState)
builder.add_node("Planner", plan_node)
builder.add_node("Executor", execute_node)
builder.add_node("RePlanner", replan_node)

builder.add_edge(START, "Planner")
builder.add_edge("Planner", "Executor")
builder.add_edge("Executor", "RePlanner")

builder.add_conditional_edges(
    "RePlanner",
    should_continue,
    {"CONTINUE": "Executor", "FINISH": END}
)

# Compile with persistence
memory = SqliteSaver(sqlite3.connect("planner.db", check_same_thread=False))
graph = builder.compile(checkpointer=memory)

# --- STEP 4: RUN ---
config = {"configurable": {"thread_id": "plan_001"}}
query = "Analyze the impact of the latest US interest rate decision on Tech stocks."

for chunk in graph.stream({"input": query, "past_steps": [], "plan": []}, config):
    print(chunk)