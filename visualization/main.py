from __future__ import annotations
import os
from datetime import datetime
import json
from pathlib import Path

from insight_extraction.categorizer.categorize import run_pipeline
from insight_extraction.semantic_intent.semantic_intent import get_semantic_intent
from models.llm_client import OpenAILLMClient
from insight_extraction.utils.saving_scripts import save_intent_to_file, load_test_intent, save_sql_results_to_csv
from insight_extraction.extraction.sql_generate import SQLQueryGenerator
from insight_extraction.extraction.extract import define_queries, extract_insights

DATA_DIR = Path("datasets")
OUT_DIR = Path("output")
USR_PROMPT_DIR = Path("initial_prompts")


def main(user_prompt: str, df_path: str, run_id: str) -> None:
    
    # dataframe structure extraction
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
    print(">>> User question:\t")
    print(user_prompt)
    print("\n>>> Calling LLM for semantic intent...\n")

    print(">>>>>>>>> -------- Intent extraction ------- <<<<<<<<<\n")
    intent = get_semantic_intent(
        user_question=user_prompt,
        llm_client=llm_client,
        schema_columns=schema_columns,
    )

    print(">>> Parsed intent JSON:")
    print(json.dumps(intent, indent=2))

    INTENT_DIR = os.path.join(OUT_DIR, "intents")
    os.makedirs(INTENT_DIR, exist_ok=True)

    intent_path = os.path.join(INTENT_DIR, f"intent_{run_id}.json")
    save_intent_to_file(intent, intent_path)
    
    print(f"\n>>> JSON salvato in: {intent_path}")

    print("\n>>>>>>>>> -------- Categorization ------- <<<<<<<<<\n")

    print("Run categorization pipeline...\n")

    allocation_path = os.path.join(OUT_DIR, f"allocation_{run_id}.json")

    run_pipeline(
        excel_path=df_path,
        intent_path=intent_path,
        output_path=allocation_path,
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.2,
        min_support_ratio=0.01,
        max_examples=None,
    )

    print(f">>> Saved file with categories allocations to: {allocation_path}\n")

    print("\n>>>>>>>>> -------- Insights extraction ------- <<<<<<<<<\n")

    DB_DIR = os.path.join(OUT_DIR, "db")
    os.makedirs(DB_DIR, exist_ok=True)

    CSV_DIR = os.path.join(OUT_DIR, "csv")
    os.makedirs(CSV_DIR, exist_ok=True)

    db_path = os.path.join(DB_DIR, f"raw_insights_{run_id}.db")
    csv_path = os.path.join(CSV_DIR, f"raw_insights_{run_id}.csv") 

    print(">>> Define and run queries to extract insights...\n")
    sql_code = define_queries(
        llm_client=llm_client,
        allocation_path=allocation_path,
        user_prompt=user_prompt,
        intent=intent,
        db_path=db_path,
        csv_path=csv_path,
    )

    INSIGHTS_DIR = os.path.join(DATA_DIR, "extracted")
    os.makedirs(INSIGHTS_DIR, exist_ok=True)

    insights_dfs = extract_insights(
        db_path=db_path,
        sql_code=sql_code,
        output_dir=INSIGHTS_DIR,
    )

    for df in insights_dfs:
        print(f"{df}\n")
    

    
    

if __name__ == "__main__":

    obs_id = 4

    prompt_path = os.path.join(USR_PROMPT_DIR, f"prompt_{obs_id}.txt")
    df_path = os.path.join(DATA_DIR, f"data_{obs_id}.xlsx")
    
    with open(prompt_path, "r") as f:
        user_prompt = f.read()
    
    
    main(user_prompt = user_prompt, df_path = df_path, run_id = obs_id)
