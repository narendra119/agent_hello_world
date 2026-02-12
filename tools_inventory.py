# Core Imports
from datetime import datetime
from typing import get_type_hints
import inspect
import json


def get_current_time() -> str:
    """A simple function to get the current time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add(a: int, b: int) -> int:
    """A simple function to add two integers"""
    return a + b


def subtract(a: int, b: int) -> int:
    """A simple function to subtract two integers"""
    return a - b


def multiply(a: int, b: int) -> int:
    """A simple function to multiply two integers"""
    return a * b


def divide(a: int, b: int) -> float:
    """A simple function to divide two integers"""
    if b == 0:
        raise ValueError("Cannot divide by zero")

    return a / b


def get_tool_description(func):
    """
    Takes a Python function and returns a dictionary
    that an LLM (Ollama/OpenAI) can consume.
    """
    # Get function name
    name = func.__name__

    # Get description from docstring
    description = func.__doc__.strip() if func.__doc__ else "No description available"

    # Get function signature
    sig = inspect.signature(func)

    # Get type hints
    try:
        type_hints = get_type_hints(func)
    except:
        type_hints = {}

    # Build parameters dictionary
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    for param_name, param in sig.parameters.items():
        # Get parameter type
        param_type = type_hints.get(param_name, str).__name__

        # Map Python types to JSON schema types
        type_mapping = {
            "int": "integer",
            "float": "number",
            "str": "string",
            "bool": "boolean",
            "list": "array",
            "dict": "object"
        }
        json_type = type_mapping.get(param_type, "string")

        parameters["properties"][param_name] = {
            "type": json_type,
            "description": f"Parameter {param_name}"
        }

        # Add to required if no default value
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(param_name)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
        }
    }

tool_functions = [
    get_current_time,
    add,
    subtract,
    multiply,
    divide
]

tool_definitions = [get_tool_description(func) for func in tool_functions]


# REGISTRY: this connect the name in the tool_defintions json to the actual function
tool_map = {
    "get_current_time": get_current_time,
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
}

# 2. THE DISPATCHER (The Logic)
def execute_tool_call(tool_call):
    name = tool_call.function.name
    args = tool_call.function.arguments  # This is already a dictionary from Ollama

    # Get the actual function from our map
    function_to_call = tool_map.get(name)

    if function_to_call:
        # Pass the arguments into the function (**kwargs unpacking)
        return function_to_call(**args)
    return "Error: Tool not found"
