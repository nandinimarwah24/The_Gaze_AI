import json
from pathlib import Path

MEMORY_FILE = Path(__file__).parent / "memory.json"


def load_memory():
    """
    Load user memory from memory.json.
    Returns a dictionary.
    """
    if not MEMORY_FILE.exists():
        save_memory({})
        return {}

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_memory(memory):
    """
    Save memory dictionary to memory.json.
    """
    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file, indent=4, ensure_ascii=False)


def update_memory(key, value):
    """
    Add or update one memory item.
    """
    memory = load_memory()
    memory[key] = value
    save_memory(memory)


def get_memory(key, default=None):
    """
    Retrieve one memory item.
    """
    memory = load_memory()
    return memory.get(key, default)


def delete_memory(key):
    """
    Delete one memory item.
    """
    memory = load_memory()

    if key in memory:
        del memory[key]
        save_memory(memory)


def clear_memory():
    """
    Remove all stored memory.
    """
    save_memory({})


def memory_exists(key):
    """
    Check if a memory key exists.
    """
    memory = load_memory()
    return key in memory


def get_all_memory():
    """
    Return the complete memory dictionary.
    """
    return load_memory()