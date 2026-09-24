from .loader import load_sleep_data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Literal, Optional

class HypnogramExtension:
    """
    Draw a hypnogram with stage bands and per-episode lines for sleep stages.

    Parameters
    ----------
    source : {'file', 'client'}
        Source of sleep data (on-disk file or in-memory dataframe).
    base_path : str or None, default None
        Base directory for file-based loading (if `source='file'`).
    subject_id : int or None, default None
        Optional subject identifier (for file-based loaders).
    filename : str or None, default None
        File name for file-based loaders.
    client_df : pandas.DataFrame or None, default None
        Preloaded sleep dataframe for `source='client'`.
    gap_threshold : int, default 30
        Maximum allowed gap (minutes) between consecutive episodes to draw
        a vertical connector (i.e., visually “stitch” transitions).

    Notes
    -----
    Expected columns in the sleep dataframe:
    - ``start`` : episode start (datetime)
    - ``end``   : episode end (datetime)
    - ``stage`` : one of {'Awake', 'REM_sleep', 'Light_sleep', 'Deep_sleep'}

    Expects the environment to provide:
    - ``load_sleep_data(...)`` function
    - a viewer with ``viewer.view_start``, ``viewer.view_end``
    - Matplotlib axis at ``self.ax``
    - ``mdates`` imported for tick formatting
    """
    def __init__(self,
                 source: Literal["file", "client"],
                 base_path: str | None = None, 
                 subject_id: int | None = None,
                 filename: str | None = None,
                 client_df: Optional[pd.DataFrame] = None,
                 gap_threshold: int = 30,
                ):
        self.source = source
        self.base_path = base_path
        self.filename = filename
        self.client_df = client_df
        self.gap_threshold = pd.Timedelta(minutes=gap_threshold)

    def draw(self):
        viewer = self.viewer
        ax = self.ax
        start, end = viewer.view_start, viewer.view_end

        # Load sleep data for this subject and restrict to current day
        sleep_df = load_sleep_data(source=self.source,
                                   base_path=self.base_path,
                                   filename=self.filename,
                                   client_df=self.client_df
                                  )
        day_sleep = sleep_df[(sleep_df["start"] < end) & (sleep_df["end"] > start)]

        # Stage mapping and background shading
        stage_map = {"Awake": 4, "REM_sleep": 3, "Light_sleep": 2, "Deep_sleep": 1}
        base_color = "#1f77b4"
        alphas = {4: 0.1, 3: 0.25, 2: 0.45, 1: 0.7}

        for val in [1, 2, 3, 4]:
            ax.axhspan(val - 0.5, val + 0.5, color=base_color, alpha=alphas[val])

        if not day_sleep.empty:
            prev_end = None
            prev_stage = None

            for _, row in day_sleep.iterrows():
                y_val = stage_map.get(row["stage"], None)
                if y_val is None:
                    continue

                seg_start = max(row["start"], start)
                seg_end = min(row["end"], end)

                ax.hlines(y_val, seg_start, seg_end, color="navy", linewidth=2)

                if prev_end is not None and (row["start"] - prev_end) <= self.gap_threshold:
                    ax.vlines(row["start"], prev_stage, y_val, color="navy", linewidth=2)

                prev_end = row["end"]
                prev_stage = y_val

        # Axis formatting
        ax.set_xlim(start, end)
        ax.set_ylim(0.5, 4.5)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(["Deep", "Light", "REM", "Awake"])
        ax.set_ylabel(
            "Sleep Stage", 
            color="0.2",
            fontsize=10
        )

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=viewer.scale(2, 6), tz=start.tzinfo))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I %p", tz=start.tzinfo))
        ax.tick_params(labelsize=viewer.scale(10, 7))
