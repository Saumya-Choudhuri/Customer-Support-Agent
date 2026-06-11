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
