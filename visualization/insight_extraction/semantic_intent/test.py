from __future__ import annotations

import json

from visualization.categorizer.semantic_intent.semantic_intent import get_semantic_intent
from visualization.model.llm_client import OpenAILLMClient
import os
from datetime import datetime



def main() -> None:
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "intent_outputs"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"intent_{timestamp}.json")
    save_intent_to_file(intent, output_path)

    print(f"\n>>> JSON salvato in: {output_path}")

    


if __name__ == "__main__":
    main()
