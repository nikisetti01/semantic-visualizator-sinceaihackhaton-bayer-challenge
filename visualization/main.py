from __future__ import annotations
import os
from datetime import datetime
import json
from pathlib import Path

from insight_extraction.categorizer.categorize import run_pipeline
from insight_extraction.semantic_intent.semantic_intent import get_semantic_intent
from models.llm_client import OpenAILLMClient
from insight_extraction.utils.saving_scripts import save_intent_to_file, load_test_intent
from insight_extraction.extraction.sql_generate import SQLQueryGenerator
from insight_extraction.extraction.table_creator import load_assignments, build_analytics_dataframe, save_dataframe_to_sqlite, save_dataframe_to_csv, save_columns_to_json



def main(timestamp: str) -> None:
    # Prompt di test (quello della challenge)
    user_question = (
        
    "Analyze all safety observations from 2024. What was the average processing time for the observations? Do any trends emerge regarding which types of observations have a longer-than-usual processing time?"
        
    )

    # Schema di esempio (puoi adattarlo a data_en)
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

    
    # Client OpenAI reale (assume OPENAI_API_KEY nell'ambiente)
    llm_client = OpenAILLMClient(
        model_name="gpt-4.1",  # o "gpt-4.1", "gpt-4o", ecc.
        temperature=0.0,
        max_output_tokens=512,
    )

    print(">>> User question:")
    print(user_question)
    print("\n>>> Calling LLM for semantic intent...\n")

    intent = get_semantic_intent(
        user_question=user_question,
        llm_client=llm_client,
        schema_columns=schema_columns,
    )

    print(">>> Parsed intent JSON:")
    print(json.dumps(intent, indent=2))

    intent_dir = f"output/intent_outputs"
    os.makedirs(intent_dir, exist_ok=True)

    intent_path = os.path.join(intent_dir, f"intent_{timestamp}.json")
    save_intent_to_file(intent, intent_path)
    
    print(f"\n>>> JSON salvato in: {intent_path}")

    print("🔍 Avvio test completo della pipeline…")

    excel_path = Path("datasets/data_en.xlsx")  # <-- MODIFICA con il tuo path

    assignement_path = Path(f"output/assignments_{timestamp}.json")

    run_pipeline(
        excel_path=excel_path,
        intent_path=intent_path,
        output_path=assignement_path,
        sheet_name="Foglio4",
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.4,
        min_support_ratio=0.01,
        max_examples=None,  # o metti un numero
    )

    print(f"🎉 Test completato. Risultati salvati in: {assignement_path}")

    # Prompt di test (quello della challenge)
    print("Generating SQL query from semantic intent...\n")

    db_path = f"output/db/analytics_{timestamp}.db"
    csv_path = f"output/csv/analytics_raw_{timestamp}.csv" 

    assignments = load_assignments(assignement_path)
    df = build_analytics_dataframe(assignments)

    save_dataframe_to_sqlite(df, db_path)
    save_dataframe_to_csv(df, csv_path)
    

    # Schema di esempio (puoi adattarlo a data_en)
    schema_text = save_columns_to_json(

        df,
        out_path="schema_columns.json"
    )

    generator = SQLQueryGenerator(llm_client=llm_client, sql_dialect="SQLite")

    sql_code = generator.generate_sql(
        user_question=user_question,
        json_spec=intent,
        table_schema_text=schema_text,
        main_table="semantic_observations",
    )

    print(sql_code)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main(timestamp)
