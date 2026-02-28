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

## MCP (Model Context Protocol)
Abstract the tool-calling logic to
connect with external MCP servers for weather, browser access, or file editing.

## SANDBOXING
Implement a 'Read-Only' flag for tools.
Ensure the agent cannot call 'write_audit_log' unless a specific environment variable is set.

## ASYNC SUPPORT
Refactor 'execute_tool_call' to be 'async def'.
This allows the agent to fetch external API data (like weather or logs) without blocking the loop.

## SEMANTIC SEARCH
Replace 'search_coding_standards' with a ChromaDB lookup.
Convert the user's query into a vector and find the most relevant document.

## MULTI-MODAL LOGGING
Create a visual dashboard (Streamlit or Flask)
that displays the 'messages' history in real-time as the agent 'thinks'.
