from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path

from insight_extraction.categorizer.categorize import run_pipeline
from insight_extraction.semantic_intent.semantic_intent import get_semantic_intent
from insight_extraction.semantic_intent.expander import expand_dimension_categories
from models.llm_client import OpenAILLMClient
from insight_extraction.utils.saving_scripts import (
    save_intent_to_file,
    save_sql_results_to_csv,
)
from insight_extraction.extraction.sql_generate import SQLQueryGenerator
from insight_extraction.extraction.table_creator import (
    load_assignments,
    build_analytics_dataframe,
    save_dataframe_to_sqlite,
    save_dataframe_to_csv,
    save_columns_to_json,
)
from insight_extraction.extraction.sql_execute import (
    execute_sql_on_sqlite,
    results_to_dataframes,
)


def main(timestamp: str) -> None:
    # ------------------------------------------------------------------
    # 1. User question (Bayer challenge)
    # ------------------------------------------------------------------
    user_question = (
        "Analyze the observations related to electrical safety from the years 2024–2025. Is there an upward or downward trend over time?"

    )

    # Schema hint per l’intent
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
    # 2. LLM clients (intent + expansions + SQL)
    # ------------------------------------------------------------------
    intent_llm = OpenAILLMClient(
        model_name="gpt-4.1",
        temperature=0.0,
        max_output_tokens=2500,
    )

    # per le espansioni serve più spazio di output
    expansion_llm = OpenAILLMClient(
        model_name="gpt-4.1",
        temperature=0.0,
        max_output_tokens=4096,
    )

    # per SQL puoi riusare intent_llm oppure crearne un altro
    sql_llm = intent_llm

    print(">>> User question:")
    print(user_question)
    print("\n>>> Calling LLM for semantic intent...\n")

    # ------------------------------------------------------------------
    # 3. Semantic intent
    # ------------------------------------------------------------------
    intent = get_semantic_intent(
        user_question=user_question,
        llm_client=intent_llm,
        schema_columns=schema_columns,
    )

    print(">>> Parsed intent JSON:")
    print(json.dumps(intent, indent=2, ensure_ascii=False))

    intent_dir = Path("output/intent_outputs")
    intent_dir.mkdir(parents=True, exist_ok=True)

    intent_path = intent_dir / f"intent_{timestamp}.json"
    save_intent_to_file(intent, str(intent_path))

    print(f"\n>>> Intent JSON salvato in: {intent_path}")

    # ------------------------------------------------------------------
    # 4. Espansione categorie (expander)
    # ------------------------------------------------------------------
    print("\n>>> Expanding semantic categories for each dimension_type...\n")

    expansions_dir = Path("output/expansion_outputs")
    expansions_dir.mkdir(parents=True, exist_ok=True)

    all_expansions: dict[str, dict[str, dict[str, any]]] = {}

    for group in intent.get("group_by", []):
        dim_type = group.get("dimension_type")
        values = list(dict.fromkeys(group.get("values", [])))  # uniq

        if not dim_type or not values:
            continue

        print(f"--- Expanding dimension: {dim_type} ({len(values)} values)")

        expanded = expand_dimension_categories(
            dimension_type=dim_type,
            values=values,
            llm_client=expansion_llm,
            extra_context=(
                "HSE domain (worker safety observations, incidents, near misses, "
                "hazards, maintenance, environmental observations)."
            ),
        )

        all_expansions[dim_type] = expanded

        exp_path = expansions_dir / f"expansion_{dim_type}_{timestamp}.json"
        with exp_path.open("w", encoding="utf-8") as f:
            json.dump(expanded, f, indent=2, ensure_ascii=False)

        print(f"Saved expansion for {dim_type} to: {exp_path}\n")

    # File unico con tutte le espansioni
    all_exp_path = expansions_dir / f"expansions_all_{timestamp}.json"
    with all_exp_path.open("w", encoding="utf-8") as f:
        json.dump(all_expansions, f, indent=2, ensure_ascii=False)

    print(f">>> All expansions saved to: {all_exp_path}\n")

    # ------------------------------------------------------------------
    # 5. Avvio pipeline di categorizzazione (con expansions_path)
    # ------------------------------------------------------------------
    print("🔍 Avvio test completo della pipeline di categorizzazione…")

    excel_path = Path("datasets/data_en.xlsx")  # <-- assicurati che il path sia corretto
    assignments_path = Path(f"output/assignments_{timestamp}.json")

    run_pipeline(
        excel_path=excel_path,
        intent_path=intent_path,
        output_path=assignments_path,
   
        model_name="all-MiniLM-L6-v2",
        expansions_path=all_exp_path,
        similarity_threshold=0.2,
        min_support_ratio=0.01,
        max_examples=None,  # o metti un numero se vuoi limitare
    )

    print(f"🎉 Test completato. Assignments salvati in: {assignments_path}")

    # ------------------------------------------------------------------
    # 6. Costruzione tabella analytics + SQLite + CSV
    # ------------------------------------------------------------------
    print("\nGenerating SQL query from semantic intent...\n")

    db_dir = Path("output/db")
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / f"analytics_{timestamp}.db"

    csv_dir = Path("output/csv")
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"analytics_raw_{timestamp}.csv"

    # Carica assignments e costruisci df analytics
    assignments = load_assignments(assignments_path)
    df = build_analytics_dataframe(assignments)

    # Salva df in SQLite e CSV
    save_dataframe_to_sqlite(df, db_path)
    save_dataframe_to_csv(df, csv_path)

    # Salva schema colonne per aiutare il generatore SQL
    schema_text = save_columns_to_json(
        df,
        out_path="schema_columns.json",
    )

    # ------------------------------------------------------------------
    # 7. Generazione query SQL dall'intent
    # ------------------------------------------------------------------
    generator = SQLQueryGenerator(llm_client=sql_llm, sql_dialect="SQLite")

    sql_code = generator.generate_sql(
        user_question=user_question,
        json_spec=intent,
        main_table="observations_enriched",
        table_schema_text=schema_text,
    )

    print(">>> Generated SQL code:\n")
    print(sql_code)
    print("\n>>> Executing SQL on SQLite DB...\n")

    # ------------------------------------------------------------------
    # 8. Esecuzione SQL + salvataggio risultati aggregati
    # ------------------------------------------------------------------
    exec_results = execute_sql_on_sqlite(db_path=str(db_path), sql_response=sql_code)

    agg_dir = Path("output/aggregate_sql_results")
    agg_dir.mkdir(parents=True, exist_ok=True)

    save_sql_results_to_csv(exec_results, output_dir=str(agg_dir))

    dfs = results_to_dataframes(exec_results)
    print(">>> SQL results as DataFrames:")
    for name, df_res in dfs.items():
        print(f"\n--- {name} ---")
        print(df_res.head())

    print("\n✅ Pipeline COMPLETA eseguita con successo.\n")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main(timestamp)
