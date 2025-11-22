import json
from pathlib import Path

def save_intent_to_file(intent: dict, output_path: str) -> None:
    """
    Salva il dizionario JSON in un file .json leggibile.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(intent, f, indent=2, ensure_ascii=False)

def load_test_intent(path: str) -> dict:
    """
    Carica un intent di test oppure un intent reale.
    Questo è solo un esempio, sostituisci con il tuo path.
    """
    intent_path = Path(path)

    if not intent_path.exists():
        raise FileNotFoundError(
            f"Intent file not found. Expected at: {intent_path}"
        )

    with intent_path.open("r", encoding="utf-8") as f:
        return json.load(f)