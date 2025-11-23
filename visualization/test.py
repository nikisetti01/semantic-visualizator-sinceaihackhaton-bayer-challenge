from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path

from insight_extraction.categorizer.categorize import run_pipeline
from insight_extraction.semantic_intent.semantic_intent import get_semantic_intent
from insight_extraction.semantic_intent.expander import (
    expand_dimension_categories,
)
from models.llm_client import OpenAILLMClient
from insight_extraction.utils.saving_scripts import (
    save_intent_to_file,
)


def main(timestamp: str) -> None:
    # ------------------------------------------------------------------
    # 1. Domanda utente (dalla challenge Bayer)
    # ------------------------------------------------------------------
    user_question = (
        "Analyze all safety observations from 2024. What was the average processing "
        "time for the observations? Do any trends emerge regarding which types of "
        "observations have a longer-than-usual processing time?"
    )

    schema_columns = [
        "Created",
        "Status",
        "Division",
        "ObservationCause",
        "Location",
        "ProcessingTimeDays",
        "ObservationType",
        "Department",
        "RiskType",
    ]

    # ------------------------------------------------------------------
    # 2. Client OpenAI reale
    # ------------------------------------------------------------------
    llm_client = OpenAILLMClient(
        model_name="gpt-4.1",
        temperature=0.0,
        max_output_tokens=2500,
    )

    print(">>> User question:")
    print(user_question)
    print("\n>>> Calling LLM for semantic intent...\n")

    # ------------------------------------------------------------------
    # 3. Semantic intent
    # ------------------------------------------------------------------
    intent = get_semantic_intent(
        user_question=user_question,
        llm_client=llm_client,
        schema_columns=schema_columns,
    )

    print(">>> Parsed intent JSON:")
    print(json.dumps(intent, indent=2, ensure_ascii=False))

    intent_dir = "output/intent_outputs"
    os.makedirs(intent_dir, exist_ok=True)

    intent_path = os.path.join(intent_dir, f"intent_{timestamp}.json")
    save_intent_to_file(intent, intent_path)

    print(f"\n>>> Intent JSON salvato in: {intent_path}")

    # ------------------------------------------------------------------
    # 4. Espansione categorie
    # ------------------------------------------------------------------
    print("\n>>> Expanding semantic categories for each dimension_type...\n")

    expansions_dir = "output/expansion_outputs"
    os.makedirs(expansions_dir, exist_ok=True)

    all_expansions = {}

    for group in intent.get("group_by", []):
        dim_type = group.get("dimension_type")
        values = list(dict.fromkeys(group.get("values", [])))

        if not dim_type or not values:
            continue

        print(f"--- Expanding dimension: {dim_type} ({len(values)} values)")

        expanded = expand_dimension_categories(
            dimension_type=dim_type,
            values=values,
            llm_client=llm_client,
            extra_context="HSE domain (worker safety observations, near misses, hazards, incidents).",
        )

        all_expansions[dim_type] = expanded

        exp_path = os.path.join(
            expansions_dir,
            f"expansion_{dim_type}_{timestamp}.json",
        )
        with open(exp_path, "w", encoding="utf-8") as f:
            json.dump(expanded, f, indent=2, ensure_ascii=False)

        print(f"Saved expansion for {dim_type} to: {exp_path}\n")

    # File unico
    all_exp_path = os.path.join(expansions_dir, f"expansions_all_{timestamp}.json")
    with open(all_exp_path, "w", encoding="utf-8") as f:
        json.dump(all_expansions, f, indent=2, ensure_ascii=False)

    print(f">>> All expansions saved to: {all_exp_path}\n")

    # ------------------------------------------------------------------
    # 5. Pipeline categorizzazione FINO A assignments.json
    # ------------------------------------------------------------------
    print("🔍 Avvio test completo della pipeline di categorizzazione…")

    excel_path = Path("datasets/data_en.xlsx")
    assignments_path = Path(f"output/assignments_{timestamp}.json")

    run_pipeline(
        excel_path=excel_path,
        intent_path=intent_path,
        output_path=assignments_path,
        sheet_name="Foglio4",
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.3,
        min_support_ratio=0.01,
        expansions_path=all_exp_path,   # <<<<<<<<<<<<<<<<<<<<< AGGIUNTO
    )

    print(f"🎉 Test completato. Assignments salvati in: {assignments_path}")
    print("👉 Fine test (nessuna parte SQL).")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main(timestamp)
