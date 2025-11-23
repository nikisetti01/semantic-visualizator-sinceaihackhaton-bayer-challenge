from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from ..prompts.intent_prompt import build_intent_prompt

# ---------------------------------------------------------------------
# Costanti: schema logico e prompt di sistema
# ---------------------------------------------------------------------



def parse_intent_response(raw_response: str) -> Dict[str, Any]:
    """
    Prova a parseare la risposta del modello come JSON.
    Se ci sono caratteri extra prima/dopo, prova a ripulire in modo conservativo.
    In produzione, qui puoi aggiungere logging e fallback più robusti.
    """
    raw = raw_response.strip()

    # Caso ideale: è già un JSON puro
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback semplice: prova ad estrarre il primo {...} completo
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = raw[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Se proprio non si riesce, solleva (in produzione potresti ritentare con un nuovo prompt)
    raise ValueError(f"LLM response is not valid JSON: {raw_response}")


# ---------------------------------------------------------------------
# Funzione principale: API call astratta verso l'LLM
# ---------------------------------------------------------------------

def get_semantic_intent(
    user_question: str,
    llm_client: Any,
) -> Dict[str, Any]:
    """
    Funzione principale del primo blocco.

    - Costruisce il prompt per il modello LLM.
    - Richiede al modello di restituire SOLO JSON con:
        raw_question, metric, time, group_by, filters, focus_topics.
    - Parse-a il JSON e lo restituisce come dict Python.

    Parametri:
    ----------
    user_question : str
        Domanda naturale dell'utente (es. "What proportion ... ?").
    llm_client : Any
        Oggetto che incapsula la chiamata al modello LLM.
        Deve avere un metodo `.invoke(prompt: str) -> str`
        oppure `.generate(prompt: str) -> str`.
        Adatta questo wrapper alla tua infrastruttura (OpenAI, Bedrock, ecc.).
    schema_columns : Optional[List[str]]
        Lista di nomi colonna disponibili nel dataframe/risultato SQL
        (es. ["Created", "Status", "Division", "ObservationCause", ...]).
        Serve come hint per rendere l'intent più aderente ai dati.

    Ritorna:
    --------
    intent : Dict[str, Any]
        Dizionario con le chiavi:
          - "raw_question"
          - "metric"
          - "time"
          - "group_by"
          - "filters"
          - "focus_topics"
    """
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
    prompt = build_intent_prompt(user_question, schema_columns)

    # Adatta questa parte alla tua implementazione LLM.
    # Esempio generico:
    if hasattr(llm_client, "invoke"):
        raw_response = llm_client.invoke(prompt)
    elif hasattr(llm_client, "generate"):
        raw_response = llm_client.generate(prompt)
    else:
        raise TypeError(
            "llm_client must expose an 'invoke(prompt: str) -> str' or "
            "'generate(prompt: str) -> str' method."
        )

    # In alcuni SDK la risposta è un oggetto complesso: qui assumiamo stringa.
    if not isinstance(raw_response, str):
        # Prova a estrarre testo dalle strutture più comuni
        try:
            raw_response = raw_response["content"]
        except Exception:
            raw_response = str(raw_response)

    intent = parse_intent_response(raw_response)
    return intent


# ---------------------------------------------------------------------
# Esempio di uso (da rimuovere o lasciare come test manuale)
# ---------------------------------------------------------------------


