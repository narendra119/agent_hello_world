# Local Imports
import time

from llm import LocalLLm
from tools_inventory import execute_tool_call, tool_definitions as local_tool_definitions
from persistence import save_messages, load_messages
from memory import store_turn, recall
from mcp_client import MCPClient
from config import LOCAL_MODEL_NAME

llm = LocalLLm(LOCAL_MODEL_NAME)

mcp = MCPClient(command="uvx", args=["mcp-server-fetch"])
mcp_fs = MCPClient(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", str(__import__("pathlib").Path(__file__).parent)])
tool_definitions = local_tool_definitions + mcp.get_tool_definitions() + mcp_fs.get_tool_definitions()
mcp_tool_names = mcp.get_tool_names() | mcp_fs.get_tool_names()

BASE_SYSTEM_CONTENT = (
    "You are a helpful assistant with access to tools. "
    "1. ONLY use a tool if the user's request explicitly requires it. "
    "2. For casual conversation, greetings, or feedback (like 'cool', 'ok', 'thanks'), "
    "do NOT call any tools. Just respond with text. "
    "3. Do NOT use tools to answer questions about your own capabilities, knowledge, or identity. Answer those directly from your own knowledge."
)


def build_system_message(memories: list) -> dict:
    content = BASE_SYSTEM_CONTENT
    if memories:
        content += "\n\nRelevant past conversations:\n" + "\n\n".join(memories)
    return {"role": "system", "content": content}


saved = load_messages()
if saved:
    print(f"Resuming previous conversation ({len(saved)} messages loaded).")
    messages = saved
else:
    messages = [build_system_message([])]

user_input = input("User: ").strip()
if user_input.lower() in ["exit", "quit"]:
    exit()
messages.append({"role": "user", "content": user_input})
messages[0] = build_system_message(recall(user_input))

iterations = 0
turn_start = time.time()

while True:
    # Step A: Call the LLM with your tool descriptions
    call_start = time.time()
    response = llm.call(messages, tools=tool_definitions)
    iterations += 1
    print(f"  [call {iterations}: {time.time() - call_start:.2f}s]")

    # Step B: Check if it wants to use a tool
    if response.message.tool_calls:
        for tool_call in response.message.tool_calls:
            # Step C: Execute the Python code
            try:
                name = tool_call.function.name
                if name in mcp_fs.get_tool_names():
                    result = mcp_fs.call_tool(name, tool_call.function.arguments)
                elif name in mcp.get_tool_names():
                    result = mcp.call_tool(name, tool_call.function.arguments)
                else:
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

    assistant_content = response.message.content
    messages.append(response.message.model_dump())
    store_turn(user_input, assistant_content)
    save_messages(messages)

    elapsed = time.time() - turn_start
    print(f"[{iterations} LLM call(s) | {elapsed:.2f}s]")

    # Take user input for the next turn and add it to the conversation history
    user_input = input("User: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append({"role": "user", "content": user_input})
    messages[0] = build_system_message(recall(user_input))
    iterations = 0
    turn_start = time.time()
