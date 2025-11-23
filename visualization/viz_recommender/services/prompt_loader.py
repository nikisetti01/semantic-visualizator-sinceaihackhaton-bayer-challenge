# services/prompt_loader.py

from pathlib import Path

DEFAULT_PROMPT_PATH = Path("initial_prompts/prompt_4.txt")

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


def load_text_file(path: str) -> str:
    """
    Generic text loader used for system prompts or user prompts.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")

    text = p.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Prompt file is empty: {p}")

    return text

