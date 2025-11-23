from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------
# Parser Robusto
# ---------------------------------------------------------------------
def parse_expansion_response(raw_response: str) -> Dict[str, Any]:
    """
    Parser robusto per la risposta LLM.

    Strategia:
    1. Tentativo diretto json.loads.
    2. Estrazione primo blocco {...}.
    3. Fix trailing commas.
    4. Salvare raw_response in un file di debug.
    """
    raw = raw_response.strip()

    # 1) Caso ideale
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2) Prova a prendere il blocco { ... }
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = raw[first_brace:last_brace + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # 3) Fix minimale per trailing comma
        fixed = (
            candidate
            .replace(",\n}", "\n}")
            .replace(",\r\n}", "\r\n}")
            .replace(", ]", " ]")
            .replace(",]", "]")
        )

        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # 4) Debug output
    debug_path = Path("expansion_debug_raw_response.json")
    try:
        debug_path.write_text(raw_response, encoding="utf-8")
    except Exception:
        pass

    raise ValueError(
        f"LLM response is not valid JSON. Raw content saved to: {debug_path}"
    )


# ---------------------------------------------------------------------
# Caricamento del prompt di sistema
# ---------------------------------------------------------------------
def load_expansion_system_prompt(prompt_path: Optional[Path] = None) -> str:
    if prompt_path is None:
        base_dir = Path(__file__).resolve().parents[1]
        prompt_path = base_dir / "prompts" / "expasion_prompt.txt"

    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------
# Costruzione del prompt
# ---------------------------------------------------------------------
def build_expansion_prompt(
    dimension_type: str,
    values: List[str],
    extra_context: Optional[str] = None,
) -> str:

    system_prompt = load_expansion_system_prompt()
    values_json = json.dumps(values, ensure_ascii=False)

    context_block = f"""
You must now expand the following dimension and its categories.

DIMENSION_TYPE:
{dimension_type}

VALUES (category names to expand):
{values_json}
"""

    if extra_context:
        context_block += f"\nADDITIONAL_CONTEXT:\n{extra_context}\n"

    reminder_block = """
Remember:
- Return ONLY a JSON object.
- JSON must contain one top-level key per category value.
- For each category, include: "name", "description", "synonyms", "examples".
- Do NOT add explanations, markdown or backticks.
"""

    return f"{system_prompt}\n\n{context_block}\n{reminder_block}".strip()


# ---------------------------------------------------------------------
# Funzione principale: espansione
# ---------------------------------------------------------------------
def expand_dimension_categories(
    dimension_type: str,
    values: List[str],
    llm_client: Any,
    extra_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Espande un dimension_type HSE e la lista dei values in un JSON semantico.
    """

    prompt = build_expansion_prompt(
        dimension_type=dimension_type,
        values=values,
        extra_context=extra_context,
    )

    # Chiamata LLM
    if hasattr(llm_client, "invoke"):
        raw_response = llm_client.invoke(prompt)
    elif hasattr(llm_client, "generate"):
        raw_response = llm_client.generate(prompt)
    else:
        raise TypeError(
            "llm_client must expose an 'invoke(prompt: str)' or 'generate(prompt: str)' method."
        )

    # Normalizzazione output
    if not isinstance(raw_response, str):
        try:
            raw_response = raw_response["content"]
        except Exception:
            raw_response = str(raw_response)

    # PARSING ROBUSTO (unico parser)
    expanded = parse_expansion_response(raw_response)

    return expanded
