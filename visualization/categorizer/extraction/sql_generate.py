from typing import Any, Dict, Optional
from ..prompts.extraction_prompt import build_extraction_prompt
from ...model.llm_client import OpenAILLMClient


class SQLQueryGenerator:
    """
    Wrapper di alto livello che usa OpenAILLMClient e build_sql_prompt
    per ottenere il codice SQL dal modello.
    """

    def __init__(
        self,
        llm_client: Optional[OpenAILLMClient] = None,
        sql_dialect: str = "SQLite",
    ) -> None:

        self.llm_client = llm_client or OpenAILLMClient()
        self.sql_dialect = sql_dialect

    def generate_sql(
        self,
        user_question: str,
        json_spec: Dict[str, Any],
        table_schema_text: str,
        main_table: str,
        data_structure_notes: Optional[str] = None,
    ) -> str:
        """
        Costruisce il prompt completo, chiama il modello e restituisce
        il codice SQL (potenzialmente più query).
        """
        prompt = build_extraction_prompt(
            user_question=user_question,
            json_spec=json_spec,
            table_schema_text=table_schema_text,
        
            sql_dialect=self.sql_dialect,
            data_structure_notes=data_structure_notes,
        )
        print(prompt)
        sql_code = self.llm_client.invoke(prompt)
        return sql_code