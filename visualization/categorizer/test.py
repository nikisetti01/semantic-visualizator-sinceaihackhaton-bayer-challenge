from __future__ import annotations

import json

from semantic_intent import get_semantic_intent
from llm_client import OpenAILLMClient
import os
from datetime import datetime

def save_intent_to_file(intent: dict, output_path: str) -> None:
    """
    Salva il dizionario JSON in un file .json leggibile.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(intent, f, indent=2, ensure_ascii=False)

def main() -> None:
    # Prompt di test (quello della challenge)
    user_question = (
        "Analyze the observations related to electrical safety from the years 2024–2025. Is there an upward or downward trend over time?"
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "intent_outputs"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"intent_{timestamp}.json")
    save_intent_to_file(intent, output_path)

    print(f"\n>>> JSON salvato in: {output_path}")

    


if __name__ == "__main__":
    main()
