import json
import os

from config import HISTORY_FILE


def save_messages(messages):
    with open(HISTORY_FILE, "w") as f:
        json.dump(messages, f, indent=2, default=str)


def load_messages():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return None
