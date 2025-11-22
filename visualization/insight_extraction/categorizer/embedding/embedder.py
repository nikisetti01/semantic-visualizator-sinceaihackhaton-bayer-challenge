import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32
) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )

def embed_categories(
    model: SentenceTransformer,
    intent: Dict[str, Any]
) -> Dict[str, Dict[str, np.ndarray]]:
    dim2cat_embs = {}
    for group in intent.get("group_by", []):
        dim = group.get("dimension_type")
        values = list(dict.fromkeys(group.get("values", [])))  # rimuove duplicati

        if values:
            cat_embs = model.encode(values, convert_to_numpy=True, normalize_embeddings=True)
            dim2cat_embs[dim] = {v: cat_embs[i] for i, v in enumerate(values)}

    return dim2cat_embs
