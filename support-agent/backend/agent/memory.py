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