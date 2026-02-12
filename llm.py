import ollama

class LocalLLm:
    def __init__(self, model):
        self.model = model

    def call(self, messages):
        response = ollama.chat(model=self.model, messages=messages)
        return response
