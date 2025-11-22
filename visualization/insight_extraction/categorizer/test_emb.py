from __future__ import annotations

import json
from pathlib import Path

from categorizer import run_pipeline

def load_test_intent() -> dict:
    """
    Carica un intent di test oppure un intent reale.
    Questo è solo un esempio, sostituisci con il tuo path.
    """
    intent_path = Path("categorizer/semantic_intent/intent_outputs/intent.json")

    if not intent_path.exists():
        raise FileNotFoundError(
            f"Intent file not found. Expected at: {intent_path}"
        )

    with intent_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("🔍 Avvio test completo della pipeline…")

    excel_path = Path("data/test_observations.xlsx")  # <-- MODIFICA con il tuo path
    intent_data = load_test_intent()
    intent_tmp_path = Path("categorizer/tests/tmp_intent.json")

    # salvo temporaneamente l'intent
    with intent_tmp_path.open("w", encoding="utf-8") as f:
        json.dump(intent_data, f, indent=2, ensure_ascii=False)

    output_path = Path("categorizer/tests/output_assignments.json")

    run_pipeline(
        excel_path=excel_path,
        intent_path=intent_tmp_path,
        output_path=output_path,
        sheet_name="Sheet1",
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.4,
        min_support_ratio=0.01,
        max_examples=None,  # o metti un numero
    )

    print(f"🎉 Test completato. Risultati salvati in: {output_path}")


if __name__ == "__main__":
    main()
