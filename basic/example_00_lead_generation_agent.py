# Load environment variables from a .env file
from dotenv import load_dotenv

# Standard library imports
import logging
import sys

# Define structured output models using Pydantic
from pydantic import BaseModel, Field

# Langchain imports that we will use to interact with Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# LangGraph imports for creating the agent
from langgraph.prebuilt import create_react_agent

# Custom tools that we will use. These are pulled from our lead_generation_tools.py
from lead_generation_tools import scrape_tool, search_tool, save_tool

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Models ---
# Pydantic models define the structure of the data we expect.
# While the agent uses tools to save data, these models serve as a good reference
# for the expected format and can be used for validation if needed.

class LeadResponse(BaseModel):
    """Model for individual lead information"""
    company: str = Field(description="The company name")
    contact_info: str = Field(description="Any available contact details")
    email: str = Field(description="Email addresses found")
    summary: str = Field(description="A brief qualification based on potential IT needs")
    outreach_message: str = Field(description="Personalized outreach message")
    tools_used: list[str] = Field(description="List of tools used to find this info")


class LeadsResponse(BaseModel):
    """Model for list of leads"""
    leads: list[LeadResponse]


class LeadGenerator:
    """Class to handle lead generation process using a ReAct agent"""
    
    def __init__(self, model_name="gemini-1.5-flash"):
        """
        Initialize the LeadGenerator with specified model.
        
        Args:
            model_name: The name of the Google Gemini model to use. 
                       (Changed default to 1.5-flash as 2.5 is likely a typo or future model)
        """
        load_dotenv()  # Load environment variables
        
        # Initialize the LLM (Large Language Model)
        # We use ChatGoogleGenerativeAI to connect to Gemini
        try:
            self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

        # List of tools the agent can use
        # The agent will decide which tool to call based on the user's request
        self.tools = [scrape_tool, search_tool, save_tool]
        
        # Create the agent using LangGraph's prebuilt create_react_agent
        # This replaces the manual chain construction in the previous version.
        # create_react_agent creates a graph that loops:
        # 1. Call LLM
        # 2. If LLM wants to call a tool -> Execute Tool -> Go back to 1
        # 3. If LLM returns text -> End
        self.agent = create_react_agent(self.llm, self.tools)
    
    def generate_leads(self, query: str = "Find and qualify exactly 5 local leads in Vancouver for IT Services. No more than 5 small businesses."):
        """
        Generate leads based on the provided query using the ReAct agent.
        
        Args:
            query: The natural language request for the agent.
        """
        logger.info(f"Starting lead generation with query: {query}")
        
        # Define the system prompt that guides the agent's behavior
        # We pass this as a SystemMessage to the agent
        system_prompt = """You are a sales enablement assistant.
            1. Use the 'search_and_scrape' tool to find information about local small businesses in Vancouver,
               British Columbia, from a variety of industries, that might need IT services.
            2. For each company identified, gather detailed information about their potential IT needs.
            3. Analyze the information to provide:
                - company: The company name
                - contact_info: Any available contact details
                - email: Email addresses
                - summary: A brief qualification based on the information, focusing on
                  their potential IT needs even if they are not an IT company.
                - outreach_message: Personalized outreach message
                - tools_used: List tools used

            4. Format the output as a JSON list of 5 entries.
            5. IMPORTANT: Use the 'save_to_text' tool to write this JSON to a file named 'leads_output.txt'.
            6. After saving, confirm to the user that the file has been created.
            """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]

        try:
            # Invoke the agent
            # stream_mode="values" allows us to see the messages as they are generated
            # but for simplicity we just invoke and get the final state
            logger.info("Invoking agent...")
            result = self.agent.invoke({"messages": messages})
            
            # The result is the final state of the graph, which contains the list of messages
            last_message = result["messages"][-1]
            logger.info(f"Agent finished. Final response: {last_message.content}")
            
            print("\n--- Process Completed ---")
            print(last_message.content)
            
        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    """Main entry point for the lead generation process"""
    # Ensure you have set GOOGLE_API_KEY in your .env file
    try:
        lead_generator = LeadGenerator()
        lead_generator.generate_leads()
    except Exception as e:
        logger.critical(f"Application failed: {e}")
