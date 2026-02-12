import ollama

class LocalLLm:
    def __init__(self, model):
        self.model = model

    def call(self, messages, tools=None):
        response = ollama.chat(
            model=self.model,
            messages=messages,
            tools=tools
        )
        return response
