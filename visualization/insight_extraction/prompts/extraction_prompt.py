import os
import json
from typing import Dict, Any, Optional
from openai import OpenAI

# Initialize OpenAI client (reads OPENAI_API_KEY from env by default)
client = OpenAI()



EXTRACTION_SYSTEM_PROMPT = """You are an expert data analyst and SQL writer.

Your job is to:
- Read: (1) the user's question, (2) a JSON spec summarizing metrics/dimensions/time windows extracted from the question, and (3) the available table schema.
- Translate these into one or more VALID {SQL_DIALECT} SELECT queries that can be run directly.
- The queries must operate ONLY on the provided schema and must not invent columns or tables.
- The primary objective is to answer the user’s explicit request as faithfully as possible.
- As a secondary objective, when helpful, you may propose 1–2 additional queries that surface relevant trends, patterns, or insights related to the user’s request (e.g. breakdowns, time trends, top-k categories). These must still be clearly connected to the original topic.

Important constraints:
- Do NOT output any natural language explanation, only SQL code (possibly multiple queries), separated by comments like: 
  -- MAIN QUERY
  -- EXTRA INSIGHT QUERY 1
- Do NOT use DDL or DML (no CREATE, INSERT, UPDATE, DELETE). Only SELECT (with CTEs, subqueries, window functions etc.).
- Use only columns and tables described in the schema the user provides.
- If the JSON spec and the user question disagree, prioritize the user question, but still follow the JSON when it does not conflict.
- When dealing with time ranges, always use the appropriate timestamp column from the schema (for example event_time) and the time window in the JSON spec.
- If multiple metrics are requested, return multiple SELECT queries, one per metric, or a single query with multiple columns, whichever is clearer.
- If a requested metric is ambiguous, choose a reasonable interpretation and encode it clearly in SQL (for example proportions as count / sum(count) over all groups).

Think through the problem carefully and design the query structure (filters, groupings, aggregations, ordering) to match the requested metrics and dimensions.
"""


def build_system_prompt(sql_dialect: str = "SQLite") -> str:
    """
    Build the system prompt that instructs the LLM how to behave
    when generating SQL.
    """
    return EXTRACTION_SYSTEM_PROMPT.format(SQL_DIALECT=sql_dialect) 


def build_user_prompt(
    user_question: str,
    json_spec: Dict[str, Any],
    table_schema_text: str,
    data_structure_notes: Optional[str] = None,
) -> str:
    """
    Build the user message that contains:
    - Original natural language question
    - JSON metric spec
    - Table schema
    - Notes on how data is structured (long/wide, meaning of columns)
    """

    if data_structure_notes is None:
        # Generic default explanation; you can override this per project.
        data_structure_notes = f"""- Each row represents an observation.
- Dimensions mentioned in the JSON spec (for example LOCATION, OBS_TYPE, STATUS, etc.) are stored in the table columns as described in the schema above.
- Time-related considerations (for example from/to dates) must use the appropriate timestamp column (for example event_time).
- If you need to aggregate observations by a dimension, you may use GROUP BY on the relevant dimension value column(s) the specific structure of the data is {table_schema_text}.
- If it makes sense, you may apply thresholds on confidence scores (for example dimension_confidence >= 0.5) when counting or aggregating, but only if that improves alignment with the requested metric."""

    json_spec_str = json.dumps(json_spec, indent=2)

    return f"""User question (natural language)
            -------------------------------
            {user_question}
            
            
            JSON metric spec (parsed from the question)
            ------------------------------------------
            This JSON summarizes the metric(s), dimensions, filters, and time considerations that should be used to answer the user question.
            
            ```json
            {json_spec_str} """

def build_extraction_prompt(
    user_question: str,
    json_spec: Dict[str, Any],
    table_schema_text: str,
    sql_dialect: str = "SQLite",
    data_structure_notes: Optional[str] = None,
) -> str:
    """
    Build the full prompt (system + user) for SQL extraction.
    """
    system_prompt = build_system_prompt(sql_dialect)
    user_prompt = build_user_prompt(
        user_question,
        json_spec,
        table_schema_text,
        data_structure_notes,
    )

    full_prompt = f"""SYSTEM PROMPT
                    {system_prompt}

                    USER PROMPT
                    {user_prompt}
                    """
    return full_prompt