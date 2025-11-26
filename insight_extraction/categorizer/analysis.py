from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_category_stats_dataframe(
    all_stats: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Converte il dict all_stats (dimension_type -> category_name -> CategoryStats-like)
    in un DataFrame comodo da ispezionare / salvare.
    """
    rows: List[Dict[str, Any]] = []

    for dim_type, cat2stats in all_stats.items():
        for cat_name, stats in cat2stats.items():
            if hasattr(stats, "__dict__"):
                s = stats.__dict__
            elif isinstance(stats, dict):
                s = stats
            else:
                s = {
                    "dimension_type": dim_type,
                    "category": cat_name,
                    "support_count": getattr(stats, "support_count", None),
                    "support_ratio": getattr(stats, "support_ratio", None),
                    "mean_score": getattr(stats, "mean_score", None),
                }

            rows.append(
                {
                    "dimension_type": s.get("dimension_type", dim_type),
                    "category": s.get("category", cat_name),
                    "support_count": s.get("support_count"),
                    "support_ratio": s.get("support_ratio"),
                    "mean_score": s.get("mean_score"),
                }
            )

    df_stats = pd.DataFrame(rows)
    return df_stats


def print_category_stats(
    all_stats: Dict[str, Dict[str, Any]],
    top_n: Optional[int] = None,
    sort_by: str = "support_count",
    ascending: bool = False,
) -> None:
    df_stats = build_category_stats_dataframe(all_stats)

    if df_stats.empty:
        print("⚠️ Nessuna statistica di categoria disponibile.")
        return

    if sort_by in df_stats.columns:
        df_stats = df_stats.sort_values(by=sort_by, ascending=ascending)

    if top_n is not None:
        df_stats = df_stats.head(top_n)

    print("\n===== CATEGORY STATS (per dimensione) =====")
    print(df_stats.to_string(index=False))
    print("==========================================\n")


def print_cluster_examples(
    df: pd.DataFrame,
    all_best_idx: Dict[str, Any],
    dim2cat_embs: Dict[str, Dict[str, Any]],
    text_col: str = "text_for_embedding",
    max_examples_per_category: int = 5,
) -> None:
    """
    Stampa per ogni dimensione e categoria un piccolo "cluster" di esempi di osservazioni.
    """
    if text_col not in df.columns:
        raise ValueError(f"La colonna '{text_col}' non esiste nel DataFrame.")

    print("\n===== CLUSTER DI ESEMPI PER CATEGORIA =====")

    for dim_type, best_idx in all_best_idx.items():
        print(f"\n--- Dimensione: {dim_type} ---")

        cat_names = list(dim2cat_embs.get(dim_type, {}).keys())
        if not cat_names:
            print("  (nessuna categoria attiva per questa dimensione)")
            continue

        best_idx_arr = np.asarray(best_idx)

        for ci, cat_name in enumerate(cat_names):
            mask = (best_idx_arr == ci)
            idx_rows = np.where(mask)[0]

            if len(idx_rows) == 0:
                continue

            print(f"\n  Categoria: {cat_name} (n={len(idx_rows)})")
            for j, row_idx in enumerate(idx_rows[:max_examples_per_category]):
                text = str(df.iloc[row_idx][text_col])
                print(f"    [{row_idx}] {text}")

    print("============================================\n")


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_category_support_bar(
    all_stats: Dict[str, Dict[str, Any]],
    dimension_type: Optional[str] = None,
    top_n: int = 10,
    normalize: bool = False,
) -> None:
    """
    Bar chart dei supporti per categoria.

    Se dimension_type è None, plottiamo tutte le dimensioni,
    altrimenti filtriamo solo quella indicata.
    """
    df_stats = build_category_stats_dataframe(all_stats)
    if df_stats.empty:
        print("⚠️ Nessuna statistica disponibile per il plotting.")
        return

    if dimension_type is not None:
        df_stats = df_stats[df_stats["dimension_type"] == dimension_type]

    if df_stats.empty:
        print(f"⚠️ Nessuna categoria per dimension_type = {dimension_type}")
        return

    # Ordina per support_count
    df_plot = df_stats.sort_values("support_count", ascending=False)

    if top_n is not None and top_n > 0:
        df_plot = df_plot.head(top_n)

    values = df_plot["support_count"].astype(float).values
    labels = df_plot["category"].astype(str).values

    if normalize and values.sum() > 0:
        values = values / values.sum()

    plt.figure(figsize=(10, max(4, 0.4 * len(labels))))
    y_pos = np.arange(len(labels))

    plt.barh(y_pos, values)
    plt.yticks(y_pos, labels)
    plt.gca().invert_yaxis()  # categorie più alte in cima

    if normalize:
        plt.xlabel("Relative support (normalized)")
    else:
        plt.xlabel("Support count")

    title = "Category support"
    if dimension_type:
        title += f" — dimension: {dimension_type}"
    plt.title(title)

    plt.tight_layout()
    plt.show()


def plot_support_vs_mean_score(
    all_stats: Dict[str, Dict[str, Any]],
    dimension_type: Optional[str] = None,
) -> None:
    """
    Scatter plot: support_count (asse X) vs mean_score (asse Y),
    un punto per categoria. Colori diversi per dimension_type.
    """
    df_stats = build_category_stats_dataframe(all_stats)
    if df_stats.empty:
        print("⚠️ Nessuna statistica disponibile per il plotting.")
        return

    if dimension_type is not None:
        df_stats = df_stats[df_stats["dimension_type"] == dimension_type]

    if df_stats.empty:
        print(f"⚠️ Nessuna categoria per dimension_type = {dimension_type}")
        return

    plt.figure(figsize=(8, 6))

    # Se nessun filtro, usiamo dimension_type per colorare
    if dimension_type is None:
        dims = df_stats["dimension_type"].unique()
        for dim in dims:
            sub = df_stats[df_stats["dimension_type"] == dim]
            plt.scatter(
                sub["support_count"],
                sub["mean_score"],
                label=str(dim),
                alpha=0.8,
            )
    else:
        plt.scatter(
            df_stats["support_count"],
            df_stats["mean_score"],
            alpha=0.8,
        )

    plt.xlabel("Support count")
    plt.ylabel("Mean similarity score")
    title = "Support vs mean score per category"
    if dimension_type:
        title += f" — {dimension_type}"
    plt.title(title)

    if dimension_type is None:
        plt.legend(title="dimension_type")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_dimension_summary(
    all_stats: Dict[str, Dict[str, Any]],
) -> None:
    """
    Summary per dimensione:
      - numero di categorie attive
      - totale support_count (sommato sulle categorie)

    Due barre affiancate per ogni dimensione.
    """
    df_stats = build_category_stats_dataframe(all_stats)
    if df_stats.empty:
        print("⚠️ Nessuna statistica disponibile per il plotting.")
        return

    agg = df_stats.groupby("dimension_type").agg(
        n_categories=("category", "nunique"),
        total_support=("support_count", "sum"),
    ).reset_index()

    dims = agg["dimension_type"].astype(str).values
    n_cat = agg["n_categories"].values
    tot_sup = agg["total_support"].values

    x = np.arange(len(dims))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, n_cat, width, label="n_categories")
    plt.bar(x + width/2, tot_sup, width, label="total_support")

    plt.xticks(x, dims, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Dimension summary: #categories and total support")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("dimension_summary.png")

    plt.close()