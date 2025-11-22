# services/file_io.py

from pathlib import Path

DEFAULT_OUTPUT_PATH = Path("chart_recommendation_1.txt")

def save_text_file(text: str) -> Path:
    """
    Save the generated text to the default output file.
    The path is internal and not passed as an argument.
    """
    path = DEFAULT_OUTPUT_PATH

    # Ensure parent folder exists
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(text, encoding="utf-8")
    return path
