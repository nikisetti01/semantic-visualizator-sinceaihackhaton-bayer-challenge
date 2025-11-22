from ..categorizer.extraction.sql_generate import SQLQueryGenerator
import json
import pandas as pd
from ...model.llm_client import OpenAILLMClient
from ..categorizer.prompts.extraction_prompt import save_columns_to_json

def main() -> None:
    # Prompt di test (quello della challenge)
    user_prompt = (
        "Analyze all safety observations from 2024. What was the average processing time for the observations? Do any trends emerge regarding which types of observations have a longer-than-usual processing time?"
    )

    # JSON spec di esempio 
    intent_path = "intent_outputs/prova.json"
    print(f"[INFO] Carico intent JSON da: {intent_path}")

    with open(intent_path, "r", encoding="utf-8") as f:
        json_intent = json.load(f)
     
    # Schema di esempio (puoi adattarlo a data_en)
    df = pd.read_csv("analytics_export.csv")
    schema_text = save_columns_to_json(
        df,
        out_path="schema_columns.json"
    )

    llm = OpenAILLMClient(
        model_name="gpt-4.1-mini",
        temperature=0.0,
        max_output_tokens=512,
    )

    generator = SQLQueryGenerator(llm_client=llm, sql_dialect="SQLite")

    sql_code = generator.generate_sql(
        user_question=user_prompt,
        json_spec=json_intent,
        table_schema_text=schema_text,
        main_table="semantic_observations",
    )

    print(sql_code)

if __name__ == "__main__":
    main()