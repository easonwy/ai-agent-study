# Self-Correcting RAG

## The Self-Correcting Flow (CRAG + Self-RAG)

In this implementation, the agent doesn't just search and answer; it grades its own search results. If the retrieved data is low quality, it automatically triggers a web search fallback and re-writes its own query.

Complexity Level: **Advanced Agentic RAG**

This demo simulates a "Deep Research Agent" that:

1. Retrieves data from a local Vector DB.
2. Grades the relevance of the documents.
3. Corrects the search path if the context is insufficient (Web Fallback)

## Why this is "Pro-Grade" Implementation

Verification Loop: Most developers build "Naive RAG" where the agent just trust whatever the search engine returns. This system treats the LLM as a verifier, significantly reducing hallucinations [1].
Modular Logic Separation: Grading, Searching, and Generating are separate nodes. You can swap llama3.1 for the qwen3.5-397b-cloud model only for the grading step to ensure maximum accuracy without slowing down the whole pipeline.
Structured Reasoning: By forcing the grader_llm to output a Pydantic object, you ensure your routing logic (yes/no) never fails due to conversational "yapping" from the local model.

## V2: How to test your Agent with this data

### Test "Local Success"

Question: "What is the project code for the QuantumVolt prototype?"
Expected Behavior: Agent should find "Silver-Bullet" in the PDF, the Grader should say "yes", and it should not search the web.

### Test "Web Fallback"

Question: "What is the current stock price of Tesla today?"
Expected Behavior: The retrieve_node will find info about batteries, but the grade_node should see it is irrelevant. It should trigger web_search_node to get the live price.

### Test "Hybrid Reasoning"

Question: "Compare the energy density of the QV-1 with the latest 2026 solid-state battery news."
Expected Behavior: The agent should retrieve the 450 Wh/kg from the PDF, but then realize it needs the "latest 2026 news" from the web to complete the comparison.
