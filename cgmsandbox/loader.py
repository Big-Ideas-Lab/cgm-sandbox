import json
import pandas as pd
from pathlib import Path
from dateutil import tz
import re
from typing import Literal, Optional

def load_cgm_data(source: Literal["file", "client"],
                  base_path: str | Path | None=None, 
                  subject_id: int | None = None, 
                  filename: str | None = None,
                  client_df: Optional[pd.DataFrame] = None
                 ) -> pd.DataFrame:

    # Prepare dataframe if given from JH client
    if source == "client":
        if client_df is None:
            raise ValueError("client_df must be provided when source='client'.")
        df = pd.DataFrame({
            "time": pd.to_datetime(client_df["effective_time_frame_date_time"], 
                                   utc=True, errors="coerce"),
            "gl": pd.to_numeric(client_df["blood_glucose_value"], errors="coerce")
        })

        return df.sort_values("time").reset_index(drop=True)
            
    # Prepare dataframe if given from local file
    base_path = Path(base_path)
    subject_dir = base_path / str(subject_id) if subject_id else base_path
    if not filename:
        raise ValueError("filename required for source='file'.")

    filename = filename.format(subject_id=subject_id)
    file_path = next((p for p in [subject_dir / filename, base_path / filename] if p.exists()), None)
    if not file_path:
        raise FileNotFoundError(f"No CGM file found (tried {filename})")

    with open(file_path, "r") as f:
        data = json.load(f)

    raw = pd.json_normalize(data.get("body", []))
    df = pd.DataFrame({
        "time": pd.to_datetime(raw["effective_time_frame.date_time"], utc=True, errors="coerce"),
        "gl": pd.to_numeric(raw["blood_glucose.value"], errors="coerce")
    })
    return df.dropna(subset=["time", "gl"]).reset_index(drop=True)


def load_sleep_data(source: Literal["file", "client"],
                    base_path: str | Path | None = None,
                    subject_id: int | None = None,
                    filename: str | None = None,
                    client_df: Optional[pd.DataFrame] = None,
                   ) -> pd.DataFrame:
    # Prepare dataframe if given from JH client
    if source == "client":
        if client_df is None:
            raise ValueError("client_df must be provided when source='client'.")

        # collect all unique episode indices
        episode_indices = sorted({
            int(m.group(1))
            for c in client_df.columns
            if (m := re.match(r"sleep_stage_episodes_(\d+)_", c))
        })

        records = []
        for _, row in client_df.iterrows():
            for i in episode_indices:
                start_col = f"sleep_stage_episodes_{i}_sleep_stage_time_frame_time_interval_start_date_time"
                end_col   = f"sleep_stage_episodes_{i}_sleep_stage_time_frame_time_interval_end_date_time"
                stage_col = f"sleep_stage_episodes_{i}_sleep_stage_state"

                start = pd.to_datetime(row[start_col], utc=True, errors="coerce")
                end   = pd.to_datetime(row[end_col], utc=True, errors="coerce")
                stage = str(row.get(stage_col, "Unknown"))

                if pd.notna(start) and pd.notna(end):
                    records.append({"start": start, "end": end, "stage": stage})

        df = pd.DataFrame(records).drop_duplicates().sort_values("start").reset_index(drop=True)
        return df

    # Prepare dataframe if given from local file
    base_path = Path(base_path)
    subject_dir = base_path / str(subject_id) if subject_id is not None else base_path

    if filename:
        filename = filename.format(subject_id=subject_id)
        candidates = [subject_dir / filename, base_path / filename]
    else:
        raise ValueError("Must specify either `filename` or `subject_id`.")
    
    json_file = next((p for p in candidates if p.exists()), None)
    if json_file is None:
        raise FileNotFoundError(f"No CGM file found (tried {candidates})")
    
    with open(json_file, "r") as file:
        data = json.load(file)
    
    records = []

    body = data.get("body", [])

    for item in body:
        episodes = item.get("sleep_stage_episodes", [])
        for ep in episodes:
            ti = ep.get("sleep_stage_time_frame", {}).get("time_interval", {})

            start = pd.to_datetime(ti.get("start_date_time"), utc=True, errors="coerce")
            end   = pd.to_datetime(ti.get("end_date_time"),   utc=True, errors="coerce")
            stage = ep.get("sleep_stage_state")

            records.append({"start": start, "end": end, "stage": stage})
    
    return pd.DataFrame(records)


def load_food_entry_data(source: Literal["file", "client"],
                         base_path: str | Path | None = None,
                         subject_id: int | None = None,
                         filename: str | None = None,
                         client_df: Optional[pd.DataFrame] = None
                        ) -> pd.DataFrame:

    # Prepare dataframe if given from JH client
    if source == "client":
        if client_df is None:
            raise ValueError("client_df must be provided when source='client'.")

        df = pd.DataFrame({
            "time": pd.to_datetime(client_df["effective_time_frame_date_time"], utc=True, errors="coerce"),
            "carbohydrate": pd.to_numeric(client_df.get("carbohydrate_value"), errors="coerce"),
            "food_name": client_df.get("food_name", "Food").astype(str).str.strip().replace({"": "food_entry"}),
            "calories": pd.to_numeric(client_df.get("calories_value"), errors="coerce")
        })
        return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        
    # Prepare dataframe if given from local file
    base_path = Path(base_path)
    subject_dir = base_path / str(subject_id) if subject_id is not None else base_path

    if filename:
        filename = filename.format(subject_id=subject_id)
        candidates = [subject_dir / filename, base_path / filename]
    else:
        raise ValueError("Must specify either `filename` or `subject_id`.")

    json_file = next((p for p in candidates if p.exists()), None)
    if json_file is None:
        raise FileNotFoundError(f"No food entry file found (tried {candidates})")

    with open(json_file, "r") as f:
        data = json.load(f)

    body = data.get("body", [])
    raw_df = pd.json_normalize(body)

    df = pd.DataFrame()
    df["time"] = pd.to_datetime(
        raw_df["effective_time_frame.date_time"], errors="coerce", utc=True
    )

    df["carbohydrate"] = pd.to_numeric(raw_df.get("carbohydrate.value"), errors="coerce")

    food_name_series = raw_df.get("food_name")
    if food_name_series is None:
        food_name_series = pd.Series(["Food"] * len(raw_df))
    df["food_name"] = food_name_series.astype(str).str.strip().replace({"": "food_entry"})

    df["calories"] = pd.to_numeric(raw_df["calories.value"], errors="coerce")

    return df