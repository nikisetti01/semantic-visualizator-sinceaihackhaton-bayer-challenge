import json

def save_intent_to_file(intent: dict, output_path: str) -> None:
    """
    Salva il dizionario JSON in un file .json leggibile.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(intent, f, indent=2, ensure_ascii=False)