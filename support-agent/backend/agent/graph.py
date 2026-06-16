import uuid
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from loguru import logger

from .memory import get_user_memories, extract_and_store_memories
from .retriever import retrieve_context
from .escalation import should_escalate, build_escalation_message

llm = OllamaLLM(model="llama3.2:3b", temperature=0.3)

class AgentState(TypedDict):
    user_id: str
    session_id: str
    message: str
    memory_context: str
    doc_context: str
    sources: list[str]
    response: str
    escalated: bool
    conversation_history: list[dict]

# ── Node 1: Load user memory ───────────────────────────────────────────────
async def load_memory(state: AgentState) -> AgentState:
    memory_ctx = await get_user_memories(state["user_id"])
    state["memory_context"] = memory_ctx
    state["memory_used"] = bool(memory_ctx)
    logger.info(f"Memory loaded for {state['user_id']}: {bool(memory_ctx)}")
    return state

# ── Node 2: Retrieve from docs ─────────────────────────────────────────────
def retrieve_docs(state: AgentState) -> AgentState:
    context, sources = retrieve_context(state["message"])
    state["doc_context"] = context
    state["sources"] = sources
    return state

# ── Node 3: Generate response ──────────────────────────────────────────────
def generate_response(state: AgentState) -> AgentState:
    # Build history string
    history_str = ""
    for turn in state["conversation_history"][-6:]:  # last 3 exchanges
        history_str += f"Customer: {turn['user']}\nAgent: {turn['agent']}\n\n"

    system_prompt = """You are a helpful, empathetic customer support agent.
Answer ONLY based on the provided documentation context.
If the answer is not in the context, say you don't have that information
and offer to connect them with a human agent.
Be concise, friendly, and professional."""

    user_prompt = f"""{system_prompt}

{state['memory_context']}

PRODUCT DOCUMENTATION:
{state['doc_context'] if state['doc_context'] else 'No relevant documentation found.'}

CONVERSATION HISTORY:
{history_str}

Customer: {state['message']}
Agent:"""

    response = llm.invoke(user_prompt)
    state["response"] = response.strip()
    return state

# ── Node 4: Check escalation ───────────────────────────────────────────────
def check_escalation(state: AgentState) -> AgentState:
    state["escalated"] = should_escalate(state["message"], state["response"])
    if state["escalated"]:
        state["response"] = build_escalation_message()
    return state

# ── Node 5: Write memory back ──────────────────────────────────────────────
async def write_memory(state: AgentState) -> AgentState:
    if len(state["conversation_history"]) > 0:
        convo_text = "\n".join([
            f"Customer: {t['user']}\nAgent: {t['agent']}"
            for t in state["conversation_history"][-4:]
        ])
        await extract_and_store_memories(
            state["user_id"], state["session_id"], convo_text
        )
    return state

# ── Build the graph ────────────────────────────────────────────────────────
def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("load_memory", load_memory)
    graph.add_node("retrieve_docs", retrieve_docs)
    graph.add_node("generate_response", generate_response)
    graph.add_node("check_escalation", check_escalation)
    graph.add_node("write_memory", write_memory)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "retrieve_docs")
    graph.add_edge("retrieve_docs", "generate_response")
    graph.add_edge("generate_response", "check_escalation")
    graph.add_edge("check_escalation", "write_memory")
    graph.add_edge("write_memory", END)

    return graph.compile()

agent = build_agent_graph()

async def run_agent(
    user_id: str,
    message: str,
    session_id: str,
    conversation_history: list[dict]
) -> dict:
    state = AgentState(
        user_id=user_id,
        session_id=session_id,
        message=message,
        memory_context="",
        doc_context="",
        sources=[],
        response="",
        escalated=False,
        conversation_history=conversation_history
    )
    result = await agent.ainvoke(state)
    return {
        "reply": result["response"],
        "sources": result["sources"],
        "escalated": result["escalated"],
        "memory_used": bool(result["memory_context"])
    }