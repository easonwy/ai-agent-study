"""
Automated Tech Blog Generator

1. Researcher Agent: Uses a search tool to find the latest facts.
2. Copywriter Agent: Takes the research and formats it into a professional post.
3. Supervisor Node: Decides if the research is sufficient or if the writing needs a rewrite


pip install langchain_ollama langgraph langchain_community duckduckgo-search langgraph-checkpoint-sqlite

"""
import sqlite3
from typing import Annotated, TypedDict, Literal
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

# ---STEP 1: MODELS & TOOLS---
llm = ChatOllama(model="qwen3.5:397b-cloud", base_url="http://localhost:11434", temperature=0)
search_tool = DuckDuckGoSearchRun()

# ---STEP 2: DEFINE THE STATE SCHEMA---
class AgentState(TypedDict):
    # This keeps track of the full conversation history
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]  # Accumulate messages
    # Internal flag to track which worker is active
    next_node: str

# ---STEP 3: WORKER AGENTS ---
# Specialist 1: The Fack Finder
researcher_agent = create_agent(model = llm,  tools=[search_tool])

# Specialist 2: The Copywriter
writer_agent = create_agent(model = llm, tools=[])


# ---STEP 4: SUPERVISOR (BRAIN) ---
class Router(BaseModel):
    """Decide the next step in the pipeline based on the current state."""
    next_step: Literal["Researcher", "Writer", "FINISH"] = Field(
         description="The next worker to call. Use 'FINISH' when the task is complete.")

supervisor_node = llm.with_structured_output(Router)

# ---STEP 5: NODE FUNCTIONS ---
def call_researcher(state: AgentState):
    print("--- DELEGATING TO RESEARCHER ---")
    response = researcher_agent.invoke(state)
    # We wrap the output so the Supervisor knows it came from the Researcher
    return {"messages": [HumanMessage(content=response["messages"][-1].content, name="Researcher")]}

def call_writer(state: AgentState):
    print("--- DELEGATING TO WRITER ---")
    response = writer_agent.invoke(state)
    return {"messages": [HumanMessage(content=response["messages"][-1].content, name="Writer")]}

def call_supervisor(state: AgentState):
    print("--- SUPERVISOR IS THINKING ---")
    system_msg = SystemMessage(content=(
        "You are a Content Manager. Your goal is to write a tech blog post. "
        "1. First, send the Researcher to find facts. "
        "2. Once facts are found, send the Writer to create the post. "
        "3. Only choose FINISH when the final blog post is high quality."
    ))
    # We pass the full history to the supervisor
    response = supervisor_node.invoke([system_msg] + state["messages"])
    return {"next_node": response.next_step}

# ---STEP 6: BUILD THE GRAPH---
builder = StateGraph(AgentState)

# Add our specialized nodes
builder.add_node("Supervisor", call_supervisor)
builder.add_node("Researcher", call_researcher)
builder.add_node("Writer", call_writer)

# The logic Flow
builder.add_edge(START, "Supervisor")

# Conditional routing from the Supervisor
builder.add_conditional_edges(
    "Supervisor",
    lambda state: state["next_node"],
    {
        "Researcher": "Researcher",
        "Writer": "Writer",
        "FINISH": END
    }
)

# Workers always report back to the Supervisor
builder.add_edge("Researcher", "Supervisor")
builder.add_edge("Writer", "Supervisor")

# ---STEP 7: PERSISTENCE & RUN---
# Setup SQLite Persistence
conn = sqlite3.connect("blog_memory.db", check_same_thread=False)
memory = SqliteSaver(conn)
graph = builder.compile(checkpointer=memory)

# Run the system
config = {"configurable": {"thread_id": "blog_post_1"}}  # Unique session ID
task = {"messages": [HumanMessage(content="Write a blog post about the latest advancements in AI.")]}

for chunk in graph.stream(task, config):
    print(chunk)
    print("-" * 30)
