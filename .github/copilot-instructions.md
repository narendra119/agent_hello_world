# AI Agent Coding Instructions - agent_hello_world

## Architecture Overview

This is a **bare-bones ReAct Agent** (Reasoning + Action) implementation that runs a synchronous loop without external frameworks. The agent chains LLM calls with tool execution to accomplish tasks.

### Core Components

1. **LLM Interface** ([llm.py](../llm.py)): Wraps Ollama API with streaming support
   - `LocalLLm.call()` streams tokens, accumulates full content, handles tool_calls
   - Uses `LOCAL_MODEL_NAME` env var (defaults to `llama3.2:3b`)
   - Returns final chunk with patched `content` and `tool_calls` fields

2. **Agent Loop** ([loop.py](../loop.py)): The main ReAct loop
   - Maintains message history as list of dicts with `role` and `content`
   - Calls LLM with tool definitions → checks for tool_calls → executes tools → appends results back
   - Messages format: `{"role": "user|assistant|tool", "content": "...", "name": "tool_name"}`
   - Tool results use role="tool" with the tool name in the message dict

3. **Tool System** ([tools_inventory.py](../tools_inventory.py)): Dynamic tool registry and dispatcher
   - `get_tool_description()`: Converts Python functions to LLM-consumable JSON schema (introspects type hints)
   - `tool_functions[]`: List of available functions (add, subtract, multiply, divide, get_system_stats, list_directory_contents, get_current_time)
   - `tool_map{}`: Registry mapping function names to actual callables
   - `execute_tool_call()`: Dispatcher that validates/coerces arguments using Pydantic, handles TypeError gracefully

## Key Patterns & Conventions

### Tool Integration
- **Tool Definition Format**: Uses OpenAI-compatible schema with `type`, `function.name`, `function.description`, `function.parameters`
- **Parameter Type Handling**: Pydantic validates and auto-casts string inputs to declared types (int, float, etc.)
- **Adding New Tools**: (1) Define function with docstring + type hints, (2) Add to `tool_functions[]`, (3) Register in `tool_map{}`
- **No Async**: Tools execute synchronously; tool calls don't block the UI but could in future refactors (see TODO comments)

### Message Protocol
- System prompt sets agent behavior: only use tools if explicitly needed, avoid tools for casual conversation
- Assistant's tool_calls are appended as `{"role": "assistant", "content": "...", "tool_calls": [...]}` (from response.message)
- Tool results are appended separately as `{"role": "tool", "content": result_string, "name": tool_name}`
- Conversation history is NOT persisted (see TODOs for persistence roadmap)

### LLM Response Handling
- `llm.call()` streams tokens to stdout immediately (except tool results)
- Tool results are printed with `print(f"Tool {name} returned: {result}")`
- First token triggers "Assistant: " prefix for visual clarity

## Development Workflow

### Setup
```bash
# Install dependencies (requires Python 3.10+)
pip install -e .

# Set up environment
echo "LOCAL_MODEL_NAME=llama3.2:3b" > .env

# Ensure Ollama service is running (localhost:11434)
ollama serve
```

### Running the Agent
```bash
python loop.py
# Interactive loop: enter user queries, agent reasons and acts
```

### Adding Functionality
1. **New Tools**: Add function to `tools_inventory.py`, update `tool_functions[]` and `tool_map{}`
2. **System Behavior**: Edit system prompt in `loop.py` messages initialization
3. **LLM Parameters**: Modify `llm.call()` arguments (streaming is already enabled)

## Important Limitations & TODOs

- **No Persistence**: Conversation history lost on restart (TODO: JSON-based conversation logger)
- **No Error Recovery**: Tool failures halt the loop (TODO: Wrap execute_tool_call in try-except, pass errors back to LLM)
- **Single-Turn Tools**: Each tool call is independent; no multi-step orchestration yet
- **No Long-Term Memory**: No vector database integration (TODO: ChromaDB/Qdrant for semantic recall)
- **No Sandboxing**: Tools have full file/system access (TODO: Read-only flag for sensitive tools like write_audit_log)

See `loop.py` comments for full roadmap (MCP, cost tracking, multi-modal UI, etc.).

## Testing & Debugging

- **Manual Testing**: Run `python loop.py` and test tool calls interactively
- **Type Validation**: Pydantic errors are caught and returned as tool results with error strings
- **Streaming**: Token-by-token output to verify agent is thinking
- **Message History**: Print `messages` dict in the loop to inspect conversation state (add debug print)

## External Dependencies

- `ollama` (>=0.6.1): Local LLM runtime and API
- `dotenv` (>=0.9.9): Environment variable loading
- `psutil` (>=7.2.2): System stats tool dependency
- `pydantic` (implicit via ollama): Type validation and coercion
- `duckduckgo-search` (>=8.1.1): Prepared for future search tool integration
