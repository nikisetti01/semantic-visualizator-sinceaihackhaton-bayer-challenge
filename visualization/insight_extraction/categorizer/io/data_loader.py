import pandas as pd

def load_observations_excel(path, title_col="Title", obs_col="Observation",
                            obs_date_col="Observation_date", proc_date_col="Processed_date",
                            sheet_name="Sheet1") -> pd.DataFrame:

    df = pd.read_excel(path, engine="openpyxl", sheet_name=sheet_name)

    df["text_for_embedding"] = (
        df[title_col].fillna("").astype(str) + " " +
        df[obs_col].fillna("").astype(str)
    ).str.strip()

    df[obs_date_col] = pd.to_datetime(df[obs_date_col], errors="coerce")
    df[proc_date_col] = pd.to_datetime(df[proc_date_col], errors="coerce")

    return df
