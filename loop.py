# Local Imports
import os

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from llm import LocalLLm
from tools_inventory import execute_tool_call, tool_definitions
from persistence import save_messages, load_messages


# TODO: PERSISTENCE - Implement a JSON-based conversation logger.
# Save 'messages' to a file after every 'Assistant' turn so you don't lose history on crash. - DONE

# TODO: DEFENSIVE DISPATCHER - Wrap 'execute_tool_call' in a try-except block.
# If a tool fails, pass the error string back to the LLM so it can try to self-correct. - DONE

# TODO: LONG-TERM MEMORY - Integrate ChromaDB or Qdrant.
# Store old conversations as embeddings to provide the agent with "Semantic Recall."

# TODO: STREAMING UI - Refactor 'llm.call' to handle streaming tokens - DONE
# so the Assistant's response feels 'alive' in the terminal or a future web UI.

# TODO: MCP (Model Context Protocol) - Abstract the tool-calling logic to
# connect with external MCP servers for weather, browser access, or file editing.

# TODO: COST/LATENCY TRACKING - Log the time taken for each LLM call
# and the number of iterations in the 'while' loop to monitor efficiency.

# TODO: SANDBOXING - Implement a 'Read-Only' flag for tools.
# Ensure the agent cannot call 'write_audit_log' unless a specific environment variable is set.

# TODO: ASYNC SUPPORT - Refactor 'execute_tool_call' to be 'async def'.
# This allows the agent to fetch external API data (like weather or logs) without blocking the loop.

# TODO: SEMANTIC SEARCH - Replace 'search_coding_standards' with a ChromaDB lookup.
# Convert the user's query into a vector and find the most relevant document.

# TODO: MULTI-MODAL LOGGING - Create a visual dashboard (Streamlit or Flask)
# that displays the 'messages' history in real-time as the agent 'thinks'.

# Env Vars
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "llama3.2:3b")  # Default to GPT-3.5 if not set


llm = LocalLLm(LOCAL_MODEL_NAME)

system_message = {
    "role": "system",
    "content": (
        "You are a helpful assistant with access to tools. "
        "1. ONLY use a tool if the user's request explicitly requires it. "
        "2. For casual conversation, greetings, or feedback (like 'cool', 'ok', 'thanks'), "
        "do NOT call any tools. Just respond with text."
    )
}

saved = load_messages()
if saved:
    print(f"Resuming previous conversation ({len(saved)} messages loaded).")
    messages = saved
else:
    messages = [system_message]

user_input = input("User: ").strip()
if user_input.lower() in ["exit", "quit"]:
    exit()
messages.append({"role": "user", "content": user_input})

while True:
    # Step A: Call the LLM with your tool descriptions
    response = llm.call(messages, tools=tool_definitions)

    # Step B: Check if it wants to use a tool
    if response.message.tool_calls:
        for tool_call in response.message.tool_calls:
            # Step C: Execute the Python code
            try:
                result = execute_tool_call(tool_call)
            except Exception as e:
                result = f"Tool error: {e}"

            # Step D: Update the conversation history
            messages.append(response.message.model_dump()) # Add the AI's intent
            messages.append({
                "role": "tool",
                "content": str(result),
                "name": tool_call.function.name
            })
            print(f"Tool {tool_call.function.name} returned: {result}")

        continue

    messages.append(response.message.model_dump())
    save_messages(messages)

    # Take user input for the next turn and add it to the conversation history
    user_input = input("User: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append({"role": "user", "content": user_input})
