from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path

from models.llm_client import OpenAILLMClient

from insight_extraction.semantic_intent.semantic_intent import get_semantic_intent
from insight_extraction.semantic_intent.expander import expand_dimension_categories

from insight_extraction.categorizer.categorize import run_pipeline

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


def main() -> None:
    # ----------------------------------------------------------------------
    # 0. Parametri di base
    # ----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Domanda di test (puoi sostituirla con quella che vuoi)
    user_question = (
        "Analyze all safety observations from 2024. "
        "What was the average processing time for the observations? "
        "Do any trends emerge regarding which types of observations "
        "have a longer-than-usual processing time?"
    )

    # Percorsi principali
    DATA_DIR = Path("datasets")
    OUT_DIR = Path("output")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    excel_path = DATA_DIR / "data_en.xlsx"    # <-- cambia se il file si chiama diversamente
    sheet_name = "Foglio4"                    # <-- cambia se necessario

    # ----------------------------------------------------------------------
    # 1. LLM client
    # ----------------------------------------------------------------------
    llm_client = OpenAILLMClient(
        model_name="gpt-4.1",    # o "gpt-4o", ecc.
        temperature=0.0,
        max_output_tokens=2048,
    )

    print(">>> USER QUESTION:")
    print(user_question)
    print("\n>>> Extracting semantic intent...\n")

    # ----------------------------------------------------------------------
    # 2. Semantic intent
    # ----------------------------------------------------------------------
    intent = get_semantic_intent(
        user_question=user_question,
        llm_client=llm_client,
        schema_columns=None,  # opzionale; puoi passare le colonne se vuoi guidarlo di più
    )

    print(">>> Parsed intent JSON:")
    print(json.dumps(intent, indent=2, ensure_ascii=False))

    intent_dir = OUT_DIR / "intent_outputs"
    intent_dir.mkdir(parents=True, exist_ok=True)

    intent_path = intent_dir / f"intent_{timestamp}.json"
    save_intent_to_file(intent, str(intent_path))
    print(f"\n>>> Intent JSON salvato in: {intent_path}\n")

    # ----------------------------------------------------------------------
    # 3. Expansion delle categorie per ciascun dimension_type
    # ----------------------------------------------------------------------
    print(">>> Expanding categories for each dimension_type...\n")

    expansions_dir = OUT_DIR / "expansion_outputs"
    expansions_dir.mkdir(parents=True, exist_ok=True)

    all_expansions: dict[str, dict[str, any]] = {}

    for group in intent.get("group_by", []):
        dim_type = group.get("dimension_type")
        values = list(dict.fromkeys(group.get("values", [])))  # uniq

        if not dim_type or not values:
            continue

        print(f"--- Expanding dimension: {dim_type} ({len(values)} values)")

        expanded = expand_dimension_categories(
            dimension_type=dim_type,
            values=values,
            llm_client=llm_client,
            extra_context=(
                "HSE domain (worker safety observations, near misses, "
                "incidents, hazards, environmental and quality issues)."
            ),
        )

        all_expansions[dim_type] = expanded

        exp_path = expansions_dir / f"expansion_{dim_type}_{timestamp}.json"
        with exp_path.open("w", encoding="utf-8") as f:
            json.dump(expanded, f, indent=2, ensure_ascii=False)

        print(f"Saved expansion for {dim_type} to: {exp_path}")

    expansions_all_path = expansions_dir / f"expansions_all_{timestamp}.json"
    with expansions_all_path.open("w", encoding="utf-8") as f:
        json.dump(all_expansions, f, indent=2, ensure_ascii=False)

    print(f"\n>>> All expansions saved to: {expansions_all_path}\n")

    # ----------------------------------------------------------------------
    # 4. Pipeline di categorizzazione (assegnazione categorie alle righe)
    # ----------------------------------------------------------------------
    print(">>> Running categorization pipeline...\n")

    assignments_path = OUT_DIR / f"assignments_{timestamp}.json"

    run_pipeline(
        excel_path=excel_path,
        intent_path=intent_path,
        output_path=assignments_path,
        sheet_name=sheet_name,
        model_name="all-MiniLM-L6-v2",
        expansions_path=expansions_all_path,
        similarity_threshold=0.2,
        min_support_ratio=0.01,
        max_examples=None,  # puoi mettere un numero per il debug rapido
    )

    print(f"🎉 Assignments salvati in: {assignments_path}\n")

    # ----------------------------------------------------------------------
    # 5. Costruzione tabella analytics e salvataggio in SQLite + CSV
    # ----------------------------------------------------------------------
    print(">>> Building analytics dataframe and saving to SQLite/CSV...\n")

    db_dir = OUT_DIR / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = OUT_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    db_path = db_dir / f"analytics_{timestamp}.db"
    analytics_csv_path = csv_dir / f"analytics_raw_{timestamp}.csv"

    assignments = load_assignments(assignments_path)
    df_analytics = build_analytics_dataframe(assignments)

    save_dataframe_to_sqlite(df_analytics, str(db_path))
    save_dataframe_to_csv(df_analytics, str(analytics_csv_path))

    print(f">>> Analytics table salvata in SQLite: {db_path}")
    print(f">>> Analytics raw CSV salvato in: {analytics_csv_path}\n")

    # ----------------------------------------------------------------------
    # 6. Salva schema colonne (per guidare l'SQL generator)
    # ----------------------------------------------------------------------
    schema_json_path = OUT_DIR / "schema_columns.json"
    schema_text = save_columns_to_json(
        df_analytics,
        out_path=str(schema_json_path),
    )
    print(f">>> Schema colonne salvato in: {schema_json_path}\n")

    # ----------------------------------------------------------------------
    # 7. Generazione SQL via LLM
    # ----------------------------------------------------------------------
    print(">>> Generating SQL queries from semantic intent...\n")

    generator = SQLQueryGenerator(llm_client=llm_client, sql_dialect="SQLite")

    # NB: main_table deve corrispondere al nome della tabella usata in SQLite
    # nei tuoi metodi: tipicamente "observations_enriched" o simile.
    # Adatta questo nome a quello effettivo creato in save_dataframe_to_sqlite.
    main_table_name = "observations_enriched"

    sql_code = generator.generate_sql(
        user_question=user_question,
        json_spec=intent,
        main_table=main_table_name,
        table_schema_text=schema_text,
    )

    print(">>> SQL generato:")
    print(sql_code)
    print("\n")

    # ----------------------------------------------------------------------
    # 8. Esecuzione SQL su SQLite e salvataggio risultati
    # ----------------------------------------------------------------------
    print(">>> Executing SQL queries on SQLite DB...\n")

    exec_results = execute_sql_on_sqlite(
        db_path=str(db_path),
        sql_response=sql_code,
    )

    agg_results_dir = OUT_DIR / "aggregate_sql_results"
    agg_results_dir.mkdir(parents=True, exist_ok=True)

    save_sql_results_to_csv(exec_results, output_dir=str(agg_results_dir))

    dfs = results_to_dataframes(exec_results)

    print(f">>> {len(dfs)} tabelle aggregate generate dalle query SQL.")
    print(f">>> CSV salvati in: {agg_results_dir}\n")

    print("Chiavi/etichette delle query eseguite:")
    for key, df in dfs.items():
        print(f" - {key}: shape={df.shape}")

    print("\n✅ Test SQL end-to-end completato.\n")


if __name__ == "__main__":
    main()
