import aiosqlite
import json
from datetime import datetime
from loguru import logger
from langchain_ollama import OllamaLLM

DB_PATH = "memory.db"

llm = OllamaLLM(model="llama3.2:3b", temperature=0)

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                created_at TEXT NOT NULL,
                session_id TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                summary TEXT
            )
        """)
        await db.commit()
    logger.info("Memory DB initialised")

async def get_user_memories(user_id: str, limit: int = 10) -> str:
    """Retrieve stored facts about a user for context injection."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT fact, category, created_at FROM memories "
            "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        ) as cursor:
            rows = await cursor.fetchall()

    if not rows:
        return ""

    facts = [f"[{row[1]}] {row[0]}" for row in rows]
    return "What I remember about this customer:\n" + "\n".join(facts)

async def extract_and_store_memories(
    user_id: str, session_id: str, conversation: str
):
    """After a conversation ends, extract memorable facts using the LLM."""
    extraction_prompt = f"""
Extract important facts about the customer from this support conversation.
Return ONLY a JSON array of objects with keys "fact" and "category".
Category must be one of: preference, issue, product, contact, complaint.
Only extract genuinely useful facts for future support sessions.
Return [] if nothing worth remembering.

Conversation:
{conversation}

JSON array only, no other text:"""

    try:
        response = llm.invoke(extraction_prompt)
        # Clean up response to extract JSON
        raw = response.strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1:
            return
        facts = json.loads(raw[start:end])

        async with aiosqlite.connect(DB_PATH) as db:
            for item in facts:
                if "fact" in item and "category" in item:
                    await db.execute(
                        "INSERT INTO memories (user_id, fact, category, created_at, session_id) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (user_id, item["fact"], item["category"],
                         datetime.utcnow().isoformat(), session_id)
                    )
            await db.commit()
        logger.info(f"Stored {len(facts)} memories for user {user_id}")
    except Exception as e:
        logger.warning(f"Memory extraction failed: {e}")

async def clear_user_memories(user_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        await db.commit()