from __future__ import annotations
import os
import json
from datetime import datetime
from pathlib import Path
import time

from models.llm_client import OpenAILLMClient

from insight_extraction.semantic_intent.semantic_intent import get_semantic_intent
from insight_extraction.semantic_intent.expander import expand_dimension_categories

from insight_extraction.categorizer.categorize import run_pipeline
from insight_extraction.utils.saving_scripts import save_intent_to_file

from insight_extraction.extraction.extract import define_queries, extract_insights


DATA_DIR = Path("datasets")
OUT_DIR = Path("output")


def profile(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        end = time.time()
        print(f"{f.__name__} executed in {(end - start)*1000:.2f} ms\n")
        return res
    return f_timer


@profile
def main(run_id: str) -> None:
    # ------------------------------------------------------------------
    # 0. USER PROMPT (messo a mano!)
    # ------------------------------------------------------------------
    user_prompt = (
       "Analyze all safety observations recorded in the last year. "
"Provide the average processing time, identify which observation types "
"tend to require more time to close, and detect any trends over time "
"(e.g., per month or per quarter).  Additionally, highlight whether certain categories such as  environmental issues, equipment hazards, unsafe acts, or near misses "
"define_queries correlate with higher processing times."
"Focus only on structured fields; qualitative interpretation will be done later."

    )

    df_path = DATA_DIR / "data_8.xlsx"   # <--- Modifica se necessario
               # <--- Modifica se necessario

    # ------------------------------------------------------------------
    # 1. LLM client
    # ------------------------------------------------------------------
    llm_client = OpenAILLMClient(
        model_name="gpt-4.1",
        temperature=0.0,
        max_output_tokens=2500,
    )

    print(">>> USER QUESTION:")
    print(user_prompt)
    print("\n>>> Extracting semantic intent...\n")

    # ------------------------------------------------------------------
    # 2. Intent extraction
    # ------------------------------------------------------------------
    intent = get_semantic_intent(
        user_question=user_prompt,
        llm_client=llm_client,
    )

    print(">>> Parsed intent JSON:")
    print(json.dumps(intent, indent=2, ensure_ascii=False))

    INTENT_DIR = OUT_DIR / "intents"
    INTENT_DIR.mkdir(parents=True, exist_ok=True)

    intent_path = INTENT_DIR / f"intent_{run_id}.json"
    save_intent_to_file(intent, str(intent_path))

    print(f"\n>>> Intent salvato in: {intent_path}\n")

    # ------------------------------------------------------------------
    # 3. Categories expansion
    # ------------------------------------------------------------------
    EXPANSIONS_DIR = OUT_DIR / "expansions"
    EXPANSIONS_DIR.mkdir(parents=True, exist_ok=True)

    all_expansions = {}

    print("\n>>> Expanding category definitions...\n")

    for group in intent.get("group_by", []):
        dim_type = group.get("dimension_type")
        values = list(dict.fromkeys(group.get("values", [])))

        if not dim_type or not values:
            continue

        print(f"- Expanding {dim_type} ({len(values)} categories)...")

        expanded = expand_dimension_categories(
            dimension_type=dim_type,
            values=values,
            llm_client=llm_client,
            extra_context="HSE domain: safety observations, incidents, hazards, environment, quality.",
        )

        all_expansions[dim_type] = expanded

        with open(EXPANSIONS_DIR / f"expansion_{dim_type}_{run_id}.json", "w", encoding="utf-8") as f:
            json.dump(expanded, f, indent=2, ensure_ascii=False)

    expansions_path = EXPANSIONS_DIR / f"expansions_all_{run_id}.json"
    with open(expansions_path, "w", encoding="utf-8") as f:
        json.dump(all_expansions, f, indent=2, ensure_ascii=False)

    print(f"\n>>> All expansions saved to: {expansions_path}\n")

    # ------------------------------------------------------------------
    # 4. Run categorization pipeline
    # ------------------------------------------------------------------
    print(">>> Running categorization pipeline...\n")

    allocation_path = OUT_DIR / f"allocation_{run_id}.json"

    run_pipeline(
        excel_path=df_path,
        intent_path=intent_path,
        output_path=allocation_path,
        model_name="all-MiniLM-L6-v2",
        expansions_path=expansions_path,
        similarity_threshold=0.2,
        min_support_ratio=0.01,
    )

    print(f">>> Allocation file saved to: {allocation_path}\n")

    # ------------------------------------------------------------------
    # 5. SQL Generation + Execution (FINO ALLE TABELLE)
    # ------------------------------------------------------------------
    print("\n>>> Generating and executing SQL queries...\n")

    DB_DIR = OUT_DIR / "db"
    DB_DIR.mkdir(parents=True, exist_ok=True)

    CSV_DIR = OUT_DIR / "csv"
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    db_path = DB_DIR / f"raw_insights_{run_id}.db"
    csv_path = CSV_DIR / f"raw_insights_{run_id}.csv"

    sql_code = define_queries(
        llm_client=llm_client,
        allocation_path=allocation_path,
        user_prompt=user_prompt,
        intent=intent,
        db_path=str(db_path),
        csv_path=str(csv_path),
    )

    print(">>> SQL Generated:")
    print(sql_code)
    print("\n")

    INSIGHTS_DIR = DATA_DIR / "extracted"
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    insights_dfs = extract_insights(
        db_path=str(db_path),
        sql_code=sql_code,
        output_dir=str(INSIGHTS_DIR),
    )

    print(f">>> {len(insights_dfs)} insight tables created:")
    for name, df in insights_dfs.items():
        print(f" - {name}: shape={df.shape}")

    print("\n✅ SQL-only test completed.\n")


if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    main(run_id)
