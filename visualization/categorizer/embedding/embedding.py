from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
@dataclass
class CategoryStats:
     dimension_type: str
     category: str
     support_count: int
     support_ratio: float
     mean_score: float


def load_observations_excel(
    path: str,
    title_col: str = "Title",
    obs_col: str = "Observation",
    obs_date_col: str = "Observation_date",
    proc_date_col: str = "Processed_date",
    sheet_name: str = "Sheet1",
) -> pd.DataFrame:
        df = pd.read_excel(path, engine="openpyxl", sheet_name=sheet_name)
        missing = [c for c in [title_col, obs_col, obs_date_col, proc_date_col] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in Excel: {missing}")

        # Costruisce il testo per embedding
        df["text_for_embedding"] = (
            df[title_col].fillna("").astype(str).str.strip() + " " +
            df[obs_col].fillna("").astype(str).str.strip()
        ).str.strip()

        # Converte le date in datetime
        df[obs_date_col] = pd.to_datetime(df[obs_date_col], errors="coerce")
        df[proc_date_col] = pd.to_datetime(df[proc_date_col], errors="coerce")

        return df
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Carica il modello di embedding SentenceTransformer.
    """
    model = SentenceTransformer(model_name)
    return model
def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
) -> np.ndarray:
    """
    Calcola gli embeddings per una lista di testi.
    """
    embeddings = model.encode(texts, 
                              batch_size=batch_size, 
                              convert_to_numpy=True, 
                              show_progress_bar=True,
                               normalize_embeddings=True)
    return np.asanyarray(embeddings)



def embed_categories(
    model: SentenceTransformer,
    intent: Dict[str, Any],
 
) ->  Dict[str, Dict[str, np.ndarray]]:
    """
    Calcola gli embeddings per le categorie.
    """
    dim2cat_embs: Dict[str, Dict[str, np.ndarray]] = {}
    for group in intent.get("group_by", []):
        dim_type = group.get("dimension_type")
        values = group.get("values", [])
        if not dim_type or not values:
            continue
        seen = set()
        uniq_values = []
        for v in values:
            if v not in seen:
                seen.add(v)
                uniq_values.append(v)
        cat_embs= model.encode(
            uniq_values,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        dim2cat_embs[dim_type] = {v: cat_embs[i] for i, v in enumerate(uniq_values)}
    return dim2cat_embs
def match_categories_for_dimension(
    dim_type: str,
    cat_embs: Dict[str, np.ndarray],
    obs_embs: np.ndarray,
    similarity_threshold: float = 0.4,
    min_support_ratio: float = 0.01,

)-> Tuple[Dict[str,CategoryStats], np.ndarray]:
    """
    Esegue il matching delle categorie per una dimensione specifica.
    Ritorna le statistiche delle categorie e gli indici delle osservazioni abbinate.
    """
    cat_names = list(cat_embs.keys())
    if not cat_names:
        return {}, np.full(len(obs_embs), -1, dtype=int)
    cat_matrix = np.stack([cat_embs[c] for c in cat_names], axis=0)  # [C, d]

    # Similarità coseno: [N, C]
    sims = cosine_similarity(obs_embs, cat_matrix)

    # Miglior categoria per riga
    best_idx = sims.argmax(axis=1)       # [N]
    best_scores = sims[np.arange(len(obs_embs)), best_idx]

    # Righe che superano la soglia di similarità
    matched_mask = best_scores >= similarity_threshold

    N = len(obs_embs)
    stats_per_cat: Dict[str, CategoryStats] = {}
    valid_mask = np.full(len(cat_names), False, dtype=bool)

    # Calcolo delle statistiche per categoria
    for ci, cname in enumerate(cat_names):
        mask_cat = (best_idx == ci) & matched_mask
        support_count = int(mask_cat.sum())
        support_ratio = support_count / N if N > 0 else 0.0

        if support_count == 0:
            mean_score = 0.0
        else:
            mean_score = float(best_scores[mask_cat].mean())

        if support_ratio >= min_support_ratio:
            stats_per_cat[cname] = CategoryStats(
                dimension_type=dim_type,
                category=cname,
                support_count=support_count,
                support_ratio=support_ratio,
                mean_score=mean_score,
            )
            valid_mask[ci] = True

    # Rimappa gli indici: se la categoria migliore non è valida o non supera
    # la soglia, assegna -1 (nessuna categoria).
    if stats_per_cat:
        valid_idx_set = {i for i, v in enumerate(valid_mask) if v}
        remapped_best_idx: List[int] = []
        for i, ci in enumerate(best_idx):
            if not matched_mask[i]:
                remapped_best_idx.append(-1)
            elif ci not in valid_idx_set:
                remapped_best_idx.append(-1)
            else:
                remapped_best_idx.append(ci)
        best_idx = np.array(remapped_best_idx, dtype=int)
    else:
        best_idx = np.full(len(obs_embs), -1, dtype=int)

    return stats_per_cat, best_idx
def match_all_dimensions(
    intent: Dict[str, Any],
    obs_embs: np.ndarray,
    model: SentenceTransformer,
    similarity_threshold: float = 0.4,
    min_support_ratio: float = 0.01,
) -> Tuple[
    Dict[str, Dict[str, CategoryStats]],
    Dict[str, np.ndarray],
    Dict[str, Dict[str, np.ndarray]]
]:
    dim2cat_embs = embed_categories(model, intent)
    all_stats: Dict[str, Dict[str, CategoryStats]] = {}
    all_best_idx: Dict[str, np.ndarray] = {}
    for dim_type, cat_embs in dim2cat_embs.items():
        stats, best_idx = match_categories_for_dimension(
            dim_type=dim_type,
            cat_embs=cat_embs,
            obs_embs=obs_embs,
            similarity_threshold=similarity_threshold,
            min_support_ratio=min_support_ratio,
        )
        all_stats[dim_type] = stats
        all_best_idx[dim_type] = best_idx

    return all_stats, all_best_idx, dim2cat_embs

def build_assignment_json(
    df: pd.DataFrame,
    all_best_idx: Dict[str, np.ndarray],
    dim2cat_embs: Dict[str, Dict[str, np.ndarray]],
    obs_date_col: str = "Observation_date",
    proc_date_col: str = "Processed_date",
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    n_rows = len(df)

    if max_examples is not None:
        n_rows = min(n_rows, max_examples)

    for i in range(n_rows):
        row = df.iloc[i]
        rec: Dict[str, Any] = {
            "row_index": int(i),
            "observation_date": row[obs_date_col].isoformat() if pd.notnull(row[obs_date_col]) else None,
            "processed_date": row[proc_date_col].isoformat() if pd.notnull(row[proc_date_col]) else None,
            "assignments": {},
        }

        for dim_type, best_idx in all_best_idx.items():
            if i >= len(best_idx):
                continue

            ci = int(best_idx[i])
            if ci == -1:
                continue  # nessuna categoria assegnata

            cat_names = list(dim2cat_embs[dim_type].keys())
            if ci < 0 or ci >= len(cat_names):
                continue

            rec["assignments"][dim_type] = cat_names[ci]

        records.append(rec)

    return records

    
def save_assignment_json(records: List[Dict[str, Any]], output_path: str) -> None:
    """
    Salva la lista di record JSON-like in un file .json leggibile.

    Parameters
    ----------
    records : List[Dict[str, Any]]
        Output di build_assignment_json.
    output_path : str
        Path del file di destinazione (es. "assignments.json").
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False) 
