import asyncio
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# --- 1. THE JUDGE SCHEMA ---
class EvaluationScore(BaseModel):
    score: int = Field(description="Score from 1 to 10")
    reasoning: str = Field(description="Explanation of why this score was given")

# --- 2. THE PROMPT VERSIONS ---
PROMPTS = {
    "v1_concise": "You are a helpful assistant. Give very short, one-sentence answers.",
    "v2_detailed": "You are a technical expert. Provide detailed, bulleted explanations with examples."
}

# --- 3. THE DATASET (Test Cases) ---
DATASET = [
    {
        "question": "What is the capital of France?",
        "expected": "Paris"
    },
    {
        "question": "Why does the sky appear blue during the day?",
        "expected": "Rayleigh scattering sends blue light toward the eyes more than longer wavelengths."
    },
    {
        "question": "List three benefits of using renewable energy.",
        "expected": "Lower emissions, improved energy security, and reduced operating costs"
    }
]

# --- 4. THE EVALUATION ENGINE ---
async def run_eval():
    llm = ChatOllama(model="deepseek-v3.2:cloud", base_url="http://localhost:11434")
    judge = ChatOllama(model="qwen3.5:397b-cloud", base_url="http://localhost:11434", format="json").with_structured_output(EvaluationScore)

    results = []

    for prompt_name, system_content in PROMPTS.items():
        print(f"\n--- Testing Prompt: {prompt_name} ---")
        
        for case in DATASET:
            # Step A: Get Agent Response
            chain = ChatPromptTemplate.from_messages([
                ("system", system_content),
                ("human", "{input}")
            ]) | llm
            
            response = await chain.ainvoke({"input": case["question"]})
            actual_output = response.content

            # Step B: Judge the Response
            judgement = await judge.ainvoke(f"""                        
                User Question: {case['question']}
                Expected Answer: {case['expected']}
                Actual Agent Output: {actual_output}

                Return only valid JSON matching this schema:
                {{
                    "score": int, 
                    "reasoning": str
                }}
                Score from 1 to 10 how well the agent performed on this task.
                Do not include any explanations or labels outside the JSON object.
            """)

            results.append({
                "version": prompt_name,
                "score": judgement.score,
                "reasoning": judgement.reasoning,
                "output": actual_output
            })

    # --- 5. REPORTING ---
    print("\n" + "="*50)
    print("FINAL EVALUATION REPORT")
    print("="*50)
    for r in results:
        print(f"[{r['version']}] Score: {r['score']}/10")
        print(f"Reason: {r['reasoning']}")
        print(f"Output: {r['output']}\n")

if __name__ == "__main__":
    asyncio.run(run_eval())