from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:3b", temperature=0)

ESCALATION_TRIGGERS = [
    "refund", "lawsuit", "legal action", "manager", "supervisor",
    "terrible", "unacceptable", "disgusting", "fraud", "scam",
    "cancel account", "delete my data", "gdpr", "i want to speak to"
]

def should_escalate(message: str, agent_response: str) -> bool:
    """Rule-based check first, then LLM-based for edge cases."""
    msg_lower = message.lower()

    # Fast rule-based check
    if any(trigger in msg_lower for trigger in ESCALATION_TRIGGERS):
        return True

    # LLM-based check for nuanced cases
    prompt = f"""Does this customer message require escalation to a human agent?
Escalate only for: billing disputes, legal threats, account security issues,
extreme anger, or requests the bot clearly cannot handle.

Customer message: "{message}"
Bot response: "{agent_response}"

Answer with exactly one word: YES or NO"""

    result = llm.invoke(prompt).strip().upper()
    return result.startswith("YES")

def build_escalation_message() -> str:
    return (
        "I understand this needs more attention than I can provide. "
        "I'm connecting you with a human support agent right now. "
        "They'll have full context of our conversation. "
        "Expected wait time: under 5 minutes."
    )