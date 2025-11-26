from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import json
import pandas as pd

# --- Import from intern modules ---
from insight_extraction.categorizer.embedding.model_loader import load_embedding_model
from insight_extraction.categorizer.my_io.save_json import save_assignment_json

from insight_extraction.categorizer.my_io.data_loader import load_observations_df
from insight_extraction.categorizer.embedding.embedder import embed_texts, embed_categories
from insight_extraction.categorizer.matching.multi_matcher import match_all_dimensions
from insight_extraction.categorizer.analysis import (
    print_category_stats,
    plot_dimension_summary,
    plot_support_vs_mean_score,
    plot_category_support_bar,
    print_cluster_examples)
def build_assignment_json(
    df: pd.DataFrame,
    all_best_idx: Dict[str, Any],
    dim2cat_embs: Dict[str, Dict[str, Any]],
    obs_date_col: str = "Observation_date",
    proc_date_col: str = "Processed_date",
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Costruisce una lista di record JSON-serializzabili con le assegnazioni
    di categoria per ogni riga del DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con le osservazioni (deve contenere le colonne data).
    all_best_idx : Dict[str, np.ndarray]
        Per ogni dimension_type, un array di indici di categoria (o -1).
    dim2cat_embs : Dict[str, Dict[str, np.ndarray]]
        Per ogni dimension_type, mappa category_name -> embedding.
        Qui usiamo solo le chiavi per risalire al nome della categoria.
    obs_date_col : str
        Nome colonna data osservazione.
    proc_date_col : str
        Nome colonna data processo.
    max_examples : Optional[int]
        Se impostato, limita il numero massimo di righe da esportare.

    Returns
    -------
    List[Dict[str, Any]]
        Lista di record pronti per essere salvati come JSON.
    """
    records: List[Dict[str, Any]] = []
    n_rows = len(df)

    if max_examples is not None:
        n_rows = min(n_rows, max_examples)

    for i in range(n_rows):
        row = df.iloc[i]

        rec: Dict[str, Any] = {
            "row_index": int(i),
            "observation_date": (
                row[obs_date_col].isoformat()
                if pd.notnull(row[obs_date_col])
                else None
            ),
            "processed_date": (
                row[proc_date_col].isoformat()
                if pd.notnull(row[proc_date_col])
                else None
            ),
            "assignments": {},
        }

        for dim_type, best_idx in all_best_idx.items():
            if i >= len(best_idx):
                continue

            ci = int(best_idx[i])
            if ci == -1:
                # nessuna categoria assegnata
                continue

            cat_names = list(dim2cat_embs[dim_type].keys())
            if ci < 0 or ci >= len(cat_names):
                continue

            rec["assignments"][dim_type] = cat_names[ci]

        records.append(rec)

    return records


def run_pipeline(
    df: pd.DataFrame,
    intent_path: str | Path,
    output_path: str | Path = "assignments.json",
    title_col: str = "Title",
    obs_col: str = "Observation",
    obs_date_col: str = "Observation_date",
    proc_date_col: str = "Processed_date",
    model_name: str = "all-MiniLM-L6-v2",
    expansions_path: Optional[str | Path] = None,
    similarity_threshold: float = 0.4,
    min_support_ratio: float = 0.01,
 
    max_examples: Optional[int] = None,
) -> None:
    """
    Esegue l'intera pipeline di categorizzazione:

    1. Carica le osservazioni da Excel.
    2. Carica l'JSON di intent / categorie.
    3. Calcola gli embedding delle osservazioni.
    4. Calcola gli embedding delle categorie.
    5. Esegue il matching per tutte le dimensioni.
    6. Costruisce i record JSON di assegnazione.
    7. Salva l'output su file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con le osservazioni.
    intent_path : str | Path
        Path al file JSON con la definizione delle categorie
        (es. struttura con "group_by").
    output_path : str | Path
        Dove salvare l'output JSON con le assegnazioni.
    sheet_name : str
        Nome del foglio Excel da caricare.
    title_col, obs_col, obs_date_col, proc_date_col : str
        Nomi delle colonne nel file Excel.
    model_name : str
        Nome del modello SentenceTransformer da usare.
    similarity_threshold : float
        Soglia minima di similarità coseno per considerare una categoria.
    min_support_ratio : float
        Rapporto minimo di supporto per mantenere una categoria.
    max_examples : Optional[int]
        Limite opzionale al numero di righe da processare.
    """

    intent_path = Path(intent_path)
    output_path = Path(output_path)
    

    # 1. Carica osservazioni
    print(f"[1/7] Carico Excel")
    df = load_observations_df(
        df=df,
        title_col=title_col,
        obs_col=obs_col,
        obs_date_col=obs_date_col,
        proc_date_col=proc_date_col,
    )

    # 2. Carica intent JSON
    print(f"[2/7] Carico intent JSON da: {intent_path}")
    with intent_path.open("r", encoding="utf-8") as f:
        intent = json.load(f)
    
    # 2b. Carico expansions se fornite
    expansions = None
    if expansions_path is not None:
        expansions_path = Path(expansions_path)
        print(f"[2b/7] Carico expansions da: {expansions_path}")
        with expansions_path.open("r", encoding="utf-8") as f:
            expansions = json.load(f)
    # 3. Carica modello
    print(f"[3/7] Carico modello di embedding: {model_name}")
    model = load_embedding_model(model_name=model_name)

    # 4. Embedding osservazioni
    print("[4/7] Calcolo embedding delle osservazioni...")
    texts = df["text_for_embedding"].tolist()
    obs_embs = embed_texts(model, texts)

    # 5. Embedding categorie
    print("[5/7] Calcolo embedding delle categorie...")
    if expansions is None:
        raise ValueError(
            "Hai chiamato embed_categories senza expansions. "
            "Devi passare expansions_path a run_pipeline()."
        )
    dim2cat_embs = embed_categories(model, intent, expansions)

    # 6. Matching per tutte le dimensioni
    print("[6/7] Eseguo il matching categorie...")
    all_stats, all_best_idx = match_all_dimensions(
        intent=intent,
        obs_embs=obs_embs,
        dim2cat_embs=dim2cat_embs,
        similarity_threshold=similarity_threshold,
        min_support_ratio=min_support_ratio,
    )
    print_category_stats(all_stats)

    # Plot di sintesi per dimensione
    plot_dimension_summary(all_stats)

    # Plot support vs mean_score
    plot_support_vs_mean_score(all_stats)  # tutte le dimensioni insieme
    # oppure per una dimensione specifica
    # plot_support_vs_mean_score(all_stats, dimension_type="OBSERVATION_TYPE")

    # Bar chart delle categorie top per una dimensione
    plot_category_support_bar(
        all_stats,
        dimension_type="OBSERVATION_TYPE",
        top_n=10,
        normalize=False,
    )

    # Cluster di testo (solo console)
    print_cluster_examples(
        df=df,
        all_best_idx=all_best_idx,
        dim2cat_embs=dim2cat_embs,
        text_col="text_for_embedding",
        max_examples_per_category=5,
    )

    # (Opzionale) puoi loggare un piccolo riepilogo delle stats
    for dim, stats in all_stats.items():
        print(f"  - Dimensione '{dim}': {len(stats)} categorie attive")

    # 7. Costruzione records + salvataggio JSON
    print("[7/7] Costruisco i record di assegnazione...")
    records = build_assignment_json(
        df=df,
        all_best_idx=all_best_idx,
        dim2cat_embs=dim2cat_embs,
        obs_date_col=obs_date_col,
        proc_date_col=proc_date_col,
        max_examples=max_examples,
    )

    print(f"Salvo {len(records)} record in: {output_path}")
    save_assignment_json(records, str(output_path))

    print("✅ Pipeline completata.")

