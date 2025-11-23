from __future__ import annotations

import json
from pathlib import Path

from insight_extraction.categorizer.categorize import run_pipeline
from insight_extraction.semantic_intent.semantic_intent import get_semantic_intent
from insight_extraction.semantic_intent.expander import expand_dimension_categories
from models.llm_client import OpenAILLMClient
from insight_extraction.utils.saving_scripts import (
    save_intent_to_file,
)
from insight_extraction.extraction.extract import define_queries, extract_insights

# Cartelle base
DATA_DIR = Path("datasets")
OUT_DIR = Path("output")
USR_PROMPT_DIR = Path("initial_prompts")


def main(user_prompt: str, df_path: str | Path, run_id: str | int) -> None:
    run_id_str = str(run_id)

    # ------------------------------------------------------------------
    # 1. Schema colonne del dataframe
    # ------------------------------------------------------------------
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
    # 2. LLM client unico (intent + expansions + SQL/queries)
    # ------------------------------------------------------------------
    llm_client = OpenAILLMClient(
        model_name="gpt-4.1",
        temperature=0.0,
        max_output_tokens=4096,  # un po' più alto così regge anche le espansioni
    )

    print(">>> User question:\t")
    print(user_prompt)
    print("\n>>> Calling LLM for semantic intent...\n")

    # ------------------------------------------------------------------
    # 3. Intent extraction
    # ------------------------------------------------------------------
    print(">>>>>>>>> -------- Intent extraction ------- <<<<<<<<<\n")
    intent = get_semantic_intent(
        user_question=user_prompt,
        llm_client=llm_client,
        schema_columns=schema_columns,
    )

    print(">>> Parsed intent JSON:")
    print(json.dumps(intent, indent=2, ensure_ascii=False))

    intent_dir = OUT_DIR / "intents"
    intent_dir.mkdir(parents=True, exist_ok=True)

    intent_path = intent_dir / f"intent_{run_id_str}.json"
    save_intent_to_file(intent, str(intent_path))

    print(f"\n>>> JSON salvato in: {intent_path}")

    # ------------------------------------------------------------------
    # 4. Espansione categorie (expander) + salvataggio
    # ------------------------------------------------------------------
    print("\n>>>>>>>>> -------- Categorization ------- <<<<<<<<<\n")
    print("Run categorization pipeline...\n")

    expansions_dir = OUT_DIR / "expansion_outputs"
    expansions_dir.mkdir(parents=True, exist_ok=True)

    all_expansions: dict[str, dict[str, dict[str, object]]] = {}

    # loop sulle dimensioni del group_by dell'intent
    for group in intent.get("group_by", []):
        dim_type = group.get("dimension_type")
        values = list(dict.fromkeys(group.get("values", [])))  # valori unici

        if not dim_type or not values:
            continue

        print(f"--- Expanding dimension: {dim_type} ({len(values)} values)")

        expanded = expand_dimension_categories(
            dimension_type=dim_type,
            values=values,
            llm_client=llm_client,
            extra_context=(
                "HSE domain (worker safety observations, incidents, near misses, "
                "hazards, maintenance, environmental observations)."
            ),
        )

        # salvataggio espansione singola dimensione
        exp_path = expansions_dir / f"expansion_{dim_type}_{run_id_str}.json"
        with exp_path.open("w", encoding="utf-8") as f:
            json.dump(expanded, f, indent=2, ensure_ascii=False)

        print(f"Saved expansion for {dim_type} to: {exp_path}\n")

        all_expansions[dim_type] = expanded

    # file cumulativo di tutte le espansioni
    if all_expansions:
        all_exp_path = expansions_dir / f"expansions_all_{run_id_str}.json"
        with all_exp_path.open("w", encoding="utf-8") as f:
            json.dump(all_expansions, f, indent=2, ensure_ascii=False)

        print(f">>> All expansions saved to: {all_exp_path}\n")
        expansions_path = all_exp_path
    else:
        print(">>> Nessuna dimensione da espandere trovata nell'intent.")
        expansions_path = None  # la pipeline può gestire il caso senza espansioni

    # ------------------------------------------------------------------
    # 5. Categorizzazione (assegnazione categorie alle raw)
    # ------------------------------------------------------------------
    allocation_path = OUT_DIR / f"allocation_{run_id_str}.json"

    run_pipeline(
        excel_path=str(df_path),
        intent_path=str(intent_path),
        output_path=str(allocation_path),
        model_name="all-MiniLM-L6-v2",
        expansions_path=str(expansions_path) if expansions_path is not None else None,
        similarity_threshold=0.2,
        min_support_ratio=0.01,
        max_examples=None,
    )

    print(f">>> Saved file with categories allocations to: {allocation_path}\n")

    # ------------------------------------------------------------------
    # 6. Insights extraction (DB, CSV, query SQL + risultati)
    # ------------------------------------------------------------------
    print("\n>>>>>>>>> -------- Insights extraction ------- <<<<<<<<<\n")

    db_dir = OUT_DIR / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = OUT_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    db_path = db_dir / f"raw_insights_{run_id_str}.db"
    csv_path = csv_dir / f"raw_insights_{run_id_str}.csv"

    print(">>> Define and run queries to extract insights...\n")
    sql_code = define_queries(
        llm_client=llm_client,
        allocation_path=str(allocation_path),
        user_prompt=user_prompt,
        intent=intent,
        db_path=str(db_path),
        csv_path=str(csv_path),
    )

    insights_dir = DATA_DIR / "extracted"
    insights_dir.mkdir(parents=True, exist_ok=True)

    insights_dfs = extract_insights(
        db_path=str(db_path),
        sql_code=sql_code,
        output_dir=str(insights_dir),
    )

    for df in insights_dfs:
        print(f"{df}\n")


if __name__ == "__main__":
    obs_id = 4

    prompt_path = USR_PROMPT_DIR / f"prompt_{obs_id}.txt"
    df_path = DATA_DIR / f"data_{obs_id}.xlsx"

    with prompt_path.open("r", encoding="utf-8") as f:
        user_prompt = f.read()

    main(user_prompt=user_prompt, df_path=df_path, run_id=obs_id)
