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
import os
from pathlib import Path
import pandas as pd
from typing import Dict, Any


def save_sql_results_to_csv(
    results: Dict[str, Dict[str, Any]],
    output_dir: str | Path = "aggregate_dataset"
) -> None:
    """
    Salva ogni risultato SQL (ottenuto da execute_sql_on_sqlite) 
    come file CSV separato nella cartella 'aggregate_dataset/'.

    Parametri:
      - results: dict ottenuto da execute_sql_on_sqlite(...)
      - output_dir: cartella di output (default: aggregate_dataset)

    Output:
      - Crea file come:
        aggregate_dataset/main_query.csv
        aggregate_dataset/extra_insight_query_1.csv
        ...
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, payload in results.items():
        cols = payload["columns"]
        rows = payload["rows"]

        df = pd.DataFrame(rows, columns=cols)

        # Nome file sicuro (senza spazi o caratteri strani)
        safe_key = key.replace(" ", "_").replace("/", "_").split(":")[0]

        out_path = output_dir / f"{safe_key}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")

        print(f"[✓] Salvato: {out_path}")

    