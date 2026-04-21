import logging
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_AGENT_REVISIONS = 2


class TriageResult(BaseModel):
    intent: Literal["billing", "technical", "shipping", "general"] = Field(
        description="The user's intent. Must be one of: 'billing', 'technical', 'shipping', or 'general'. For order tracking, use 'shipping'."
    )
    sentiment: Literal["happy", "neutral", "frustrated", "angry"] = Field(
        description="The user's sentiment. Must be one of: 'happy', 'neutral', 'frustrated', or 'angry'."
    )


class ToneReview(BaseModel):
    is_approved: bool = Field(
        description="Must be true if the response is empathetic and appropriate, false otherwise."
    )
    critique: str = Field(description="Feedback if rejected. Leave empty if approved.")


class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    sentiment: str
    approved: bool
    critique: str
    revision_count: int


class SupportEngine:
    def __init__(self):
        self.llm = ChatOllama(model="kimi-k2.5:cloud", temperature=0)
        self.triage_llm = self.llm.with_structured_output(TriageResult)
        self.critic_llm = self.llm.with_structured_output(ToneReview)

    async def get_graph(self, mcp_client, checkpointer):
        tools = await mcp_client.get_tools()
        tool_names = [tool.name for tool in tools] if isinstance(tools, list) else []
        logger.info("[Graph] Loaded support tools: %s", tool_names or "none")

        def triage_node(state: SupportState):
            logger.info("[Graph] Step 1/3 - Triage started.")
            sys_msg = SystemMessage(
                content=(
                    "You are an expert customer support triage agent. Analyze the user's message, "
                    "extract the intent and sentiment, and output ONLY valid JSON matching the schema. "
                    "IMPORTANT: intent MUST be exactly 'billing', 'technical', 'shipping', or 'general'. "
                    "sentiment MUST be exactly 'happy', 'neutral', 'frustrated', or 'angry'."
                )
            )
            messages = [sys_msg] + state["messages"]
            result = self.triage_llm.invoke(messages)
            logger.info(
                "[Graph] Step 1/3 - Triage complete. intent=%s sentiment=%s",
                result.intent,
                result.sentiment,
            )
            return {
                "intent": result.intent,
                "sentiment": result.sentiment,
                "approved": False,
                "critique": "",
                "revision_count": 0,
            }

        agent = create_agent(
            self.llm,
            tools=tools,
            system_prompt=(
                "You are a Support Specialist. Be empathetic, concise, and action-oriented. "
                "Use tools for order-related lookups when needed. "
                "If critique feedback is present in the conversation, revise the answer to address it."
            ),
        )

        async def agent_node(state: SupportState):
            attempt = state.get("revision_count", 0) + 1
            logger.info(
                "[Graph] Step 2/3 - Agent attempt %d started. intent=%s sentiment=%s critique_present=%s",
                attempt,
                state.get("intent"),
                state.get("sentiment"),
                bool(state.get("critique")),
            )

            agent_input = dict(state)
            messages = list(state.get("messages", []))
            critique = state.get("critique", "").strip()
            if critique:
                messages.append(
                    SystemMessage(
                        content=(
                            "Revise your previous response using this tone feedback: "
                            f"{critique}. Keep the answer helpful and do not mention the critique explicitly."
                        )
                    )
                )
            agent_input["messages"] = messages

            result = await agent.ainvoke(agent_input)
            response_messages = result.get("messages", [])
            last_ai_message = next(
                (message for message in reversed(response_messages) if isinstance(message, AIMessage)),
                None,
            )
            preview = (
                str(last_ai_message.content)[:100].replace("\n", " ")
                if last_ai_message
                else "no AI message returned"
            )
            logger.info("[Graph] Step 2/3 - Agent attempt %d complete. %s", attempt, preview)

            result["revision_count"] = attempt
            result["approved"] = False
            return result

        def critic_node(state: SupportState):
            logger.info("[Graph] Step 3/3 - Critic review started.")
            sys_msg = SystemMessage(
                content=(
                    "You are a tone review critic. Evaluate the response for empathy based on the user's "
                    "sentiment. You must output ONLY valid JSON matching the schema EXACTLY with properties: "
                    "'is_approved' (boolean) and 'critique' (string)."
                )
            )
            user_msg = HumanMessage(
                content=f"Sentiment: {state['sentiment']}\nResponse: {state['messages'][-1].content}"
            )
            review = self.critic_llm.invoke([sys_msg, user_msg])

            logger.info(
                "[Graph] Step 3/3 - Critic complete. approved=%s critique=%s",
                review.is_approved,
                review.critique or "<empty>",
            )
            return {"approved": review.is_approved, "critique": review.critique}

        def route_after_critic(state: SupportState):
            if state["approved"]:
                logger.info("[Graph] Critic approved response. Finishing workflow.")
                return "finish"

            if state.get("revision_count", 0) >= MAX_AGENT_REVISIONS:
                logger.warning(
                    "[Graph] Critic rejected response after %d attempt(s). Ending to avoid infinite retry loop.",
                    state.get("revision_count", 0),
                )
                return "finish"

            logger.info(
                "[Graph] Critic requested revision. Routing back to Agent with critique for attempt %d.",
                state.get("revision_count", 0) + 1,
            )
            return "Agent"

        builder = StateGraph(SupportState)
        builder.add_node("Triage", triage_node)
        builder.add_node("Agent", agent_node)
        builder.add_node("Critic", critic_node)

        builder.add_edge(START, "Triage")
        builder.add_edge("Triage", "Agent")
        builder.add_edge("Agent", "Critic")
        builder.add_conditional_edges(
            "Critic",
            route_after_critic,
            {"finish": END, "Agent": "Agent"},
        )

        logger.info("[Graph] Support workflow compiled successfully.")
        return builder.compile(checkpointer=checkpointer)
