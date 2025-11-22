from __future__ import annotations
import json

from embedding import (
    load_observations_excel,
    load_embedding_model,
    embed_texts,
    match_all_dimensions,
    build_assignment_json,
)


def main() -> None:
    # ---------------------------------------------------------------
    # 1) Carica INTENT JSON da intent_outputs/prova.json
    # ---------------------------------------------------------------
    intent_path = "intent_outputs/prova.json"
    print(f"[INFO] Carico intent JSON da: {intent_path}")

    with open(intent_path, "r", encoding="utf-8") as f:
        intent = json.load(f)

    print("[INFO] Intent JSON caricato correttamente.")

    # ---------------------------------------------------------------
    # 2) Carica dataset da Excel
    # ---------------------------------------------------------------
    excel_path = "../datasets/data_en.xlsx"  # cambia se necessario
    print(f"[INFO] Carico dataset da: {excel_path}")

    df_obs = load_observations_excel(
        path=excel_path,
        title_col="Title",
        obs_col="Observation",
        obs_date_col="Observation_date",
        proc_date_col="Processed_date",
        sheet_name="Foglio4",
    )

    print(f"[INFO] Numero righe dataset: {len(df_obs)}")

    # ---------------------------------------------------------------
    # 3) Carica modello HuggingFace (consigliato: all-mpnet-base-v2)
    # ---------------------------------------------------------------
    model_name = "sentence-transformers/all-mpnet-base-v2"
    print(f"[INFO] Carico modello embedding: {model_name}")
    model = load_embedding_model(model_name=model_name)

    # ---------------------------------------------------------------
    # 4) Embedding del dataset
    # ---------------------------------------------------------------
    print("[INFO] Embedding delle osservazioni...")
    obs_embs = embed_texts(
        model=model,
        texts=df_obs["text_for_embedding"].tolist(),
        batch_size=64
    )
    print(f"[INFO] Embedding shape: {obs_embs.shape}")

    # ---------------------------------------------------------------
    # 5) Matching semantico per tutte le dimensioni del group_by
    # ---------------------------------------------------------------
    print("[INFO] Eseguo matching semantico...")
    similarity_threshold = 0.20
    min_support_ratio = 0.01

    all_stats, all_best_idx, dim2cat_embs = match_all_dimensions(
        intent=intent,
        obs_embs=obs_embs,
        model=model,
        similarity_threshold=similarity_threshold,
        min_support_ratio=min_support_ratio,
    )

    # ---------------------------------------------------------------
    # 6) Riepilogo categorie mantenute
    # ---------------------------------------------------------------
    print("\n[INFO] Risultati matching per dimensione:")
    for dim_type, stats in all_stats.items():
        print(f"\n  Dimensione: {dim_type}")
        if not stats:
            print("    (nessuna categoria valida — tutte scartate)")
            continue

        for cname, s in stats.items():
            print(
                f"    - {cname:25s} | "
                f"support={s.support_count:4d} | "
                f"ratio={s.support_ratio:.3f} | "
                f"mean_score={s.mean_score:.3f}"
            )

    # ---------------------------------------------------------------
    # 7) Generazione JSON assegnazioni
    # ---------------------------------------------------------------
    print("\n[INFO] Costruisco JSON di assegnazione per le prime 50 righe...")

    assignment_records = build_assignment_json(
        df=df_obs,
        all_best_idx=all_best_idx,
        dim2cat_embs=dim2cat_embs,
        max_examples=50,
    )

    output_path = "assignments_test.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(assignment_records, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Assegnazioni salvate in: {output_path}")

    # ---------------------------------------------------------------
    # 8) Stampa 5 esempi
    # ---------------------------------------------------------------
    print("\n[INFO] Esempi (prime 5 righe):\n")
    for rec in assignment_records[:5]:
        print(json.dumps(rec, indent=2))
        print("-" * 60)


if __name__ == "__main__":
    main()
