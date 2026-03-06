# CLAUDE.md — agent_hello_world

## Project Overview

A bare-bones ReAct (Reasoning + Action) agent built from scratch in Python — no LangChain, no agent frameworks. The agent is a `while True` loop that calls a local LLM, dispatches tool calls, and maintains both short-term (context window) and long-term (vector DB) memory.

## Architecture

```
loop.py           — Main agent loop (ReAct: reason → act → observe → repeat)
llm.py            — Ollama streaming wrapper (LocalLLm)
tools_inventory.py — Tool definitions + dispatcher (Pydantic-validated)
mcp_client.py     — MCP server client (fetch + filesystem servers)
memory.py         — Long-term memory: store/recall via Qdrant + Ollama embeddings
persistence.py    — Short-term persistence: conversation history to JSON
vector_db.py      — Qdrant retrieval helpers (used by search_cat_facts)
config.py         — Central config (reads .env, sets all tunable constants)
```

## Stack

- **Runtime**: Python 3.10+, managed with `uv`
- **LLM**: [Ollama](https://ollama.ai) (local), default model `llama3.2:3b`
- **Embeddings**: `nomic-embed-text` via Ollama
- **Vector DB**: Qdrant (must be running locally on port 6333)
- **MCP**: `mcp-server-fetch` (web fetch) + `@modelcontextprotocol/server-filesystem` (file access)
- **Tool validation**: Pydantic (auto-coerces LLM string args to typed Python)

## Running the Agent

**Prerequisites:**
- Ollama running with `llama3.2:3b` and `nomic-embed-text` pulled
- Qdrant running locally (`docker run -p 6333:6333 qdrant/qdrant`)
- Node.js available (for the MCP filesystem server via `npx`)

```bash
uv run loop.py
```

Type `exit` or `quit` to end the session.

## Configuration

All settings live in [config.py](config.py) and can be overridden via `.env`:

| Variable             | Default                    | Description                     |
|----------------------|----------------------------|---------------------------------|
| `LOCAL_MODEL_NAME`   | `llama3.2:3b`              | Ollama chat model               |
| `EMBED_MODEL`        | `nomic-embed-text`         | Ollama embedding model          |
| `VECTOR_SIZE`        | `768`                      | Must match the embedding model  |
| `MEMORY_TOP_K`       | `3`                        | Number of past turns to recall  |
| `QDRANT_HOST`        | `localhost`                | Qdrant host                     |
| `QDRANT_PORT`        | `6333`                     | Qdrant port                     |
| `QDRANT_COLLECTION`  | `agent_memory`             | Qdrant collection name          |
| `HISTORY_FILE`       | `conversation_history.json`| Short-term persistence file     |

## Local Tools

Defined in [tools_inventory.py](tools_inventory.py):

- `get_system_stats` — CPU/memory/OS info via psutil
- `get_current_time` — current datetime
- `add`, `subtract`, `multiply`, `divide` — arithmetic
- `search_cat_facts` — semantic search over a cat-facts vector store

Tool schemas are auto-generated from Python type hints and docstrings. The dispatcher uses Pydantic to validate and coerce LLM-provided arguments.

## MCP Tools

Two MCP servers are started automatically on launch (via `mcp_client.py`):

- `mcp-server-fetch` — lets the agent fetch URLs
- `@modelcontextprotocol/server-filesystem` — lets the agent read/write files in the project directory

## Key Conventions

- **Do not use LangChain or other agent frameworks.** The point of this project is to understand agent internals.
- **All config belongs in `config.py`.** Do not hardcode model names, hosts, ports, or file paths elsewhere.
- **Tool functions must have full type hints and a docstring.** The dispatcher and schema generator depend on both.
- **MCP tool names must not collide with local tool names.** The dispatcher routes by name — conflicts will silently shadow tools.
- **Conversation history is a plain list of dicts** following the Ollama message format (`role`, `content`, optionally `tool_calls`). Keep it that way.

## Pending Work (see TODO.md)

- Sandboxing: read-only flag, confirm before side-effecting tools (HUMAN-IN-THE-LOOP)
- Async support: refactor `execute_tool_call` to `async def`
- Context window management: trim/summarize old messages before hitting the model limit
- Retry logic: exponential backoff around `llm.call`
- Token counting: track per-call token usage
- Conversation reset command (`reset`/`clear` in chat)
- Structured logging: replace `print()` with proper log levels + timestamps
- Multi-modal dashboard: Streamlit/Flask UI for real-time message inspection
