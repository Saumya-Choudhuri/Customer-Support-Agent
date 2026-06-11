import aiosqlite
import json
from datetime import datetime
from loguru import logger
from langchain_ollama import OllamaLLM

DB_PATH = "memory.db"