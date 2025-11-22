# services/prompt_loader.py

from pathlib import Path

DEFAULT_PROMPT_PATH = Path("initial_prompt/prompt_1.txt")

def load_user_query() -> str:
    """
    Load the user query from the default file
    """
    if not DEFAULT_PROMPT_PATH.exists():
        raise FileNotFoundError(f"User prompt file not found: {DEFAULT_PROMPT_PATH}")

    content = DEFAULT_PROMPT_PATH.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError(f"User prompt file is empty: {DEFAULT_PROMPT_PATH}")

    return content
