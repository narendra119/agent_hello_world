import ollama

class LocalLLm:
    def __init__(self, model):
        self.model = model

    def call(self, messages, tools=None):
        stream = ollama.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=True,
        )

        full_content = ""
        tool_calls = None
        first_token = True
        final_chunk = None

        for chunk in stream:
            token = chunk.message.content or ""
            if token:
                if first_token:
                    print("Assistant: ", end="", flush=True)
                    first_token = False
                print(token, end="", flush=True)
                full_content += token
            if chunk.message.tool_calls:
                tool_calls = chunk.message.tool_calls
            final_chunk = chunk

        if not first_token:
            print()  # newline after streamed content

        # Patch accumulated state back onto the final chunk for the caller
        if full_content:
            final_chunk.message.content = full_content
        if tool_calls:
            final_chunk.message.tool_calls = tool_calls

        return final_chunk
