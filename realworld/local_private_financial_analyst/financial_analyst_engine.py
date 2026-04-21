from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain.agents import create_agent 
from langchain_core.tools import tool

# --- 1. The Math Tool ---
import textwrap

@tool
def calculate_math(expr: str) -> str:
    """
    Use this for precise financial calculations (totals, percentages, taxes).
    Input can be Python code with multiple statements separated by semicolons or newlines.
    The final computed value will be returned.
    """
    try:
        # Remove common leading whitespace and convert semicolons to newlines
        code = textwrap.dedent(expr).replace(';', '\n').strip()
        
        # If it's a simple expression (no assignments and simple newlines), use eval
        if '=' not in code and '\n' not in code:
            return str(eval(code))
        
        # For statements with assignments, use exec and capture the result
        namespace = {}
        exec(code, namespace)
        
        # Return the last assigned or computed value
        # Look backwards through the namespace for the last meaningful variable
        for key in reversed(list(namespace.keys())):
            if not key.startswith('_'):
                return str(namespace[key])
        
        return "Calculation completed"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

# --- 2. STATE DEFINITION ---
class AnalystState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- 3. CORE LOGIC CLASS ---
class FinancialAnalystEngine:
    def __init__(self):
        self.llm = ChatOllama(model="qwen3.5:397b-cloud", base_url="http://localhost:11434", temperature=0)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        # Initialize VectorStore
        self.vectorstore = Chroma(persist_directory="./finance_vector_db", embedding_function=self.embeddings, collection_name="finance_docs")
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.client = None

    def get_rag_tool(self, query: str) -> str:
        """
        Use this tool to get financial documents related to the query.
        Input should be a financial query.
        """
        docs = self.retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    async def get_agent(self, checkpointer, client):
        mcp_tools = await client.get_tools()

        # Combine tools
        all_tools = [calculate_math, self.get_rag_tool] + mcp_tools

        instruction = """You are a Private Financial Analyst.
        1. Use 'query_transactions' for SQL data.
        2. Use 'get_rag_tool' for PDF data.
        3. ALWAYS use 'calculate_math' for final calculations to ensure precision.
        4. If a source is empty, check the other before giving up."""

        agent = create_agent(
            model=self.llm,
            tools= all_tools,
            system_prompt=instruction,
            checkpointer=checkpointer,
            interrupt_before=["tools"]
        )
        return agent, client
        