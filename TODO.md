# TODO

## PERSISTENCE - DONE
Implement a JSON-based conversation logger.
Save 'messages' to a file after every 'Assistant' turn so you don't lose history on crash.

## DEFENSIVE DISPATCHER - DONE
Wrap 'execute_tool_call' in a try-except block.
If a tool fails, pass the error string back to the LLM so it can try to self-correct.

## LONG-TERM MEMORY - DONE
Integrate ChromaDB or Qdrant.
Store old conversations as embeddings to provide the agent with "Semantic Recall."

## STREAMING UI - DONE
Refactor 'llm.call' to handle streaming tokens
so the Assistant's response feels 'alive' in the terminal or a future web UI.

## COST/LATENCY TRACKING - DONE
Log the time taken for each LLM call
and the number of iterations in the 'while' loop to monitor efficiency.

## MCP (Model Context Protocol) - DONE
Abstract the tool-calling logic to
connect with external MCP servers for weather, browser access, or file editing.

## SANDBOXING
Implement a 'Read-Only' flag for tools.
Ensure the agent cannot call 'write_audit_log' unless a specific environment variable is set.

## ASYNC SUPPORT
Refactor 'execute_tool_call' to be 'async def'.
This allows the agent to fetch external API data (like weather or logs) without blocking the loop.

## SEMANTIC SEARCH - DONE
Replace 'search_coding_standards' with a ChromaDB lookup.
Convert the user's query into a vector and find the most relevant document.

## MULTI-MODAL LOGGING
Create a visual dashboard (Streamlit or Flask)
that displays the 'messages' history in real-time as the agent 'thinks'.

## CONTEXT WINDOW MANAGEMENT
The 'messages' list grows unboundedly. Once it exceeds the model's context limit, the agent crashes.
Trim or summarize old messages from the active window to keep it within safe limits.

## RETRY LOGIC
If 'llm.call' fails (network blip, model timeout), the agent dies immediately.
Add exponential backoff with a configurable number of retries around the LLM call.

## TOKEN COUNTING
Latency is tracked but not token usage per call.
Without this, there's no early warning before hitting the context limit.

## CONVERSATION RESET
The only way to start fresh is to manually delete 'conversation_history.json'.
Support a 'reset' or 'clear' command in the chat to wipe the active session cleanly.

## STRUCTURED LOGGING
All output is print() statements mixed with user-facing text.
Add a proper log file with timestamps and log levels for easier debugging.

## CONFIG FILE - DONE
Model name, top_k, history file path, Qdrant host are scattered across multiple files.
Centralize all settings into a single config.py or .env file.

## HUMAN-IN-THE-LOOP
Before executing tools that have side effects (writes, deletes), pause and ask for confirmation.
Pairs naturally with SANDBOXING.
