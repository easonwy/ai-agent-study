import sqlite3
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# --- 1. THE OPTIMIZER SCHEMA ---
class PromptImprovement(BaseModel):
    new_constraint: str = Field(description="A new rule to add to the system prompt to avoid previous mistakes.")

# --- 2. THE AGENT CLASS ---
class SelfImprovingAgent:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.1:8b", temperature=0)
        self.optimizer = ChatOllama(model="qwen3.5:397b-cloud").with_structured_output(PromptImprovement)
        self.base_prompt = "You are a helpful assistant."
        self.learned_constraints = [] # This would be saved in SQLite in production

    def get_system_prompt(self):
        constraints = "\n".join([f"- {c}" for c in self.learned_constraints])
        return f"{self.base_prompt}\n\nAdditional Rules learned from feedback:\n{constraints}"

    def chat(self, user_input):
        messages = [SystemMessage(content=self.get_system_prompt()), HumanMessage(content=user_input)]
        response = self.llm.invoke(messages)
        return response.content

    def learn_from_feedback(self, last_input, last_output, feedback_text):
        print("--- 🧠 AGENT IS LEARNING FROM CRITICISM ---")
        # The Optimizer analyzes the failure
        improvement = self.optimizer.invoke(f"""
            The user disliked this interaction.
            User Input: {last_input}
            Agent Output: {last_output}
            User Feedback: {feedback_text}
            
            Identify one specific rule to add to the system prompt to prevent this error.
        """)
        self.learned_constraints.append(improvement.new_constraint)
        print(f"✅ New Rule Added: {improvement.new_constraint}")

# --- 3. THE DEMO ---
agent = SelfImprovingAgent()

# Interaction 1: Agent fails a preference
print("User: Explain quantum computing like I'm 5.")
res = agent.chat("Explain quantum computing like I'm 5.")
print(f"Agent: {res}")

# Feedback: User thinks it was too long
print("\n[User gives 👎: 'Too many big words and too long!']")
agent.learn_from_feedback(
    "Explain quantum computing like I'm 5.",
    res,
    "Too many big words and too long! Use a simple metaphor about cats."
)

# Interaction 2: Agent has improved!
print("\nUser: Explain black holes.")
res_improved = agent.chat("Explain black holes.")
print(f"Agent (Improved): {res_improved}")