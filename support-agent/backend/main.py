import uuid
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from loguru import logger

from agent.memory import init_db
from agent.graph import run_agent
from models.schemas import ChatMessage, ChatResponse

# In-memory session store (use Redis in production)
sessions: dict[str, list[dict]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("Support Agent API started")
    yield
    logger.info("Shutting down")

app = FastAPI(title="Support Agent API", lifespan=lifespan)

@app.post("/chat", response_model=ChatResponse)
async def chat(msg: ChatMessage):
    session_id = msg.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    result = await run_agent(
        user_id=msg.user_id,
        message=msg.message,
        session_id=session_id,
        conversation_history=history
    )

    # Append to session history
    history.append({"user": msg.message, "agent": result["reply"]})

    return ChatResponse(
        reply=result["reply"],
        session_id=session_id,
        sources=result["sources"],
        escalated=result["escalated"],
        memory_used=result["memory_used"]
    )

@app.delete("/memory/{user_id}")
async def clear_memory(user_id: str):
    from agent.memory import clear_user_memories
    await clear_user_memories(user_id)
    return {"status": "cleared", "user_id": user_id}

@app.get("/health")
def health():
    return {"status": "ok"}

# Serve frontend
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
def root():
    return FileResponse("../frontend/index.html")