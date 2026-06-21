# Customer-Support-Agent memory
A RAG-Powered AI Aupport agent thet remembrs customers across sessions.
Build with 100% free, local tools - no paid API keys required.

##Quick Start (GitHub Codepsaces)
1. Open this repo in a Codespaces (it auto-runs setup)
2. Wait for Ollama models to download(~2-3 GB total, one-time)
3. Ingest smaple docs:
```bash
   cd backend && payhton ''//scripts/ingest_docs.py
```
4.Start the server:
```bash
   cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
5. Open the forwarded port 8000 in your browser

## Add your own docs
Drop `.md`, `.txt`, or `.pdf` files into `backend/sample_docs/`
then re-run `python scripts/ingest_docs.py`.

## How it works
- **Memory**: After each conversation, the LLM extracts key facts about
  the customer and stores them in SQLite. Next session, those facts are
  injected into the system prompt automatically.
- **RAG**: Every message is embedded and matched against your docs in
  ChromaDB. Only relevant chunks are passed to the LLM.
- **Escalation**: Rule-based triggers + LLM check decide when to hand
  off to a human agent.