"""Generate synthetic sample data in JupyterHealth Exchange (JHE) client-frame shape.

Why this exists
---------------
The sandbox's ``source="client"`` loaders expect the DataFrame returned by
``JupyterHealthClient.list_observations_df()``. That requires a live Exchange
connection, a token, and a real study, none of which are available to someone
who just cloned this repo. This script produces a small, deterministic,
credential-free dataset with the same *shape* as the real thing so the JHE demo
can be run end to end offline.

Fidelity
--------
Column names, units, and the point-vs-interval timestamp split mirror the real
"CGM & Wearables Demo" study (JHE study_id 30006):

* ``code`` carries the enum value, e.g. ``omh:blood-glucose:4.0``
* ``subject_reference`` is ``Patient/<id>``
* every ``*date_time`` field is UTC, with a naive local twin ``*_local``
* blood glucose is **point-in-time** at 15-minute cadence in ``mg/dL``
* sleep/heart-rate/oxygen/step data are **interval** records
* sleep durations are in **seconds**
* sleep-stage-summary carries **aggregate durations only** -- study 30006 does
  NOT contain ``sleep_stage_episodes_*`` columns. A separate, clearly labelled
  ``sleep_stage_episodes.csv`` is emitted so the hypnogram can be demonstrated,
  but it is beyond the real schema; see README.

Administrative columns are a representative subset of what the Exchange emits
(no ``identifier_*``, ``meta_lastUpdated``, or ``external_datasheets_*``), which
keeps the CSVs readable. The columns the loaders actually consume are complete.

Usage
-----
    python tools/make_synthetic_jhe.py [--out sample_data/jhe] [--days 7]

The output is deterministic for a given seed, so re-running produces no diff.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260924
SUBJECT_ID = 90001
SUBJECT_REF = f"Patient/{SUBJECT_ID}"
TZ_UTC = "UTC"
TZ_LOCAL = "America/New_York"

# Modality metadata: code -> (schema namespace, name, version, modality)
MODALITIES = {
    "blood_glucose": ("omh", "blood-glucose", "4.0", "sensed"),
    "heart_rate": ("omh", "heart-rate", "2.0", "sensed"),
    "oxygen_saturation": ("omh", "oxygen-saturation", "2.0", "sensed"),
    "sleep_stage_summary": ("ieee", "sleep-stage-summary", "1.0", "sensed"),
    "food_entry": ("ieee", "food-entry", "0.1", "self-reported"),
}

CODES = {
    "blood_glucose": "omh:blood-glucose:4.0",
    "heart_rate": "omh:heart-rate:2.0",
    "oxygen_saturation": "omh:oxygen-saturation:2.0",
    "sleep_stage_summary": "ieee:sleep-stage-summary:1.0",
    "food_entry": "ieee:food-entry:0.1",
}


def _next_ids(n: int) -> list[int]:
    """Sequential synthetic resource ids."""
    return list(range(70001, 70001 + n))


def _admin_columns(modality: str, n: int, offset: int, event_times: pd.Series) -> dict:
    """Shared administrative columns, mirroring tidy_observation() output."""
    ns, name, version, mode = MODALITIES[modality]
    # creation time is when the record was ingested: shortly after the event
    creation = pd.to_datetime(event_times, utc=True) + pd.Timedelta(minutes=7)
    return {
        "resource_type": CODES[modality],
        "code": CODES[modality],
        "resourceType": "Observation",
        "id": _next_ids(n) if offset == 0 else list(range(70001 + offset, 70001 + offset + n)),
        "subject_reference": SUBJECT_REF,
        "status": "final",
        "uuid": [str(uuid.UUID(int=(SEED + offset + i) % (2**128))) for i in range(n)],
        "modality": mode,
        "schema_id_name": name,
        "schema_id_version": version,
        "schema_id_namespace": ns,
        "creation_date_time": creation.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "creation_date_time_local": creation.dt.tz_convert(TZ_LOCAL)
        .dt.tz_localize(None)
        .dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_creation_date_time": pd.to_datetime(event_times, utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%S+00:00"
        ),
        "source_creation_date_time_local": pd.to_datetime(event_times, utc=True)
        .dt.tz_convert(TZ_LOCAL)
        .dt.tz_localize(None)
        .dt.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _utc_str(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _local_str(ts: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(ts, utc=True)
        .dt.tz_convert(TZ_LOCAL)
        .dt.tz_localize(None)
        .dt.strftime("%Y-%m-%dT%H:%M:%S")
    )


# --------------------------------------------------------------------------
# physiology
# --------------------------------------------------------------------------

MEALS = [  # (hour, minute, name, carbs_g, kcal)
    (8, 0, "oatmeal with berries", 48.0, 320.0),
    (13, 0, "chicken rice bowl", 72.0, 640.0),
    (19, 0, "pasta with vegetables", 80.0, 720.0),
]
SNACK = (16, 0, "greek yogurt", 14.0, 130.0)  # on some days


def glucose_series(times: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Synthetic interstitial glucose: stable overnight, post-meal excursions."""
    t_utc = pd.to_datetime(times, utc=True)
    t_local = t_utc.dt.tz_convert(TZ_LOCAL)
    minutes = t_local.dt.hour * 60 + t_local.dt.minute
    hours = minutes / 60.0

    # gentle circadian baseline: lower overnight, slightly higher evening
    g = 99.0 + 6.0 * np.sin((hours - 4.0) / 24.0 * 2 * np.pi)

    days = sorted({d.date() for d in t_local})
    for day in days:
        day_mask = t_local.dt.date == day
        for hour, minute, _, carbs, _ in MEALS:
            meal_min = hour * 60 + minute
            dt_min = minutes - meal_min
            active = day_mask & (dt_min > 0) & (dt_min < 300)
            # ~45 min to peak, close to baseline by ~3 h -- standard postprandial shape
            rise = 42.0
            amp = 1.40 * carbs
            x = dt_min[active] / rise
            g[active.values] += amp * x * np.exp(1.0 - x)
        # afternoon snack on ~half the days
        if rng.random() < 0.5:
            meal_min = SNACK[0] * 60 + SNACK[1]
            dt_min = minutes - meal_min
            active = day_mask & (dt_min > 0) & (dt_min < 180)
            x = dt_min[active] / 30.0
            g[active.values] += 1.10 * SNACK[3] * x * np.exp(1.0 - x)

    g = g + rng.normal(0, 3.2, len(g))

    # an occasional mild overnight dip (dawn-phenomenon-free, not hypoglycaemia)
    overnight = (hours < 6) | (hours > 23)
    dip = overnight & (rng.random(len(g)) < 0.012)
    g[dip.values] -= rng.uniform(18, 30, dip.sum())

    return np.clip(g, 58.0, 305.0)


def night_windows(local_days: list, rng: np.random.Generator) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """One main sleep window per night, in LOCAL wall-clock time.

    Bedtime jitter only -- no cumulative drift. Returned in UTC, which is what
    every ``*_date_time`` column in the JHE frame carries.
    """
    out = []
    for day in local_days:
        base = pd.Timestamp(day, tz=TZ_LOCAL)
        bedtime = (
            base + pd.Timedelta(hours=22, minutes=15) + pd.Timedelta(minutes=int(rng.normal(0, 28)))
        ).tz_convert(TZ_UTC)
        wake = (
            base + pd.Timedelta(days=1, hours=6, minutes=45) + pd.Timedelta(minutes=int(rng.normal(0, 26)))
        ).tz_convert(TZ_UTC)
        if wake - bedtime < pd.Timedelta(hours=5, minutes=30):
            wake = bedtime + pd.Timedelta(hours=6, minutes=15)
        out.append((bedtime, wake))
    return out


# --------------------------------------------------------------------------
# per-modality builders
# --------------------------------------------------------------------------


def build_blood_glucose(start: pd.Timestamp, days: int, rng: np.random.Generator) -> pd.DataFrame:
    times = pd.date_range(start, periods=days * 96, freq="15min", tz=TZ_UTC)
    values = np.round(glucose_series(pd.Series(times), rng)).astype(int)

    df = pd.DataFrame(_admin_columns("blood_glucose", len(times), 0, pd.Series(times)))
    df["blood_glucose_value"] = values
    df["blood_glucose_unit"] = "mg/dL"
    df["temporal_relationship_to_meal"] = "unknown"
    df["effective_time_frame_date_time"] = _utc_str(pd.Series(times)).values
    df["effective_time_frame_date_time_local"] = _local_str(pd.Series(times)).values

    # a realistic 3-hour sensor gap in the middle of the record
    gap_start = start + pd.Timedelta(days=days // 2, hours=2)
    gap_mask = (pd.to_datetime(df["effective_time_frame_date_time"], utc=True) >= gap_start) & (
        pd.to_datetime(df["effective_time_frame_date_time"], utc=True)
        < gap_start + pd.Timedelta(hours=3)
    )
    return df.loc[~gap_mask].reset_index(drop=True)


def build_sleep_stage_summary(nights, rng: np.random.Generator) -> pd.DataFrame:
    """One row per night, matching seed_rich_demo.py's ieee:sleep-stage-summary:1.0 body.

    The real body is exactly:
        sleep_stage_summary.{total_sleep_time, light_sleep_duration,
                            deep_sleep_duration, rem_sleep_duration,
                            sleep_efficiency_percentage}
        effective_time_frame.time_interval.{start,end}_date_time
        is_main_sleep
    Durations are integer seconds (see seed_rich_demo._dur).
    """
    rows = []
    for bedtime, wake in nights:
        in_bed_s = (wake - bedtime).total_seconds()
        efficiency = rng.uniform(0.86, 0.945)
        total_sleep = int(round(in_bed_s * efficiency))
        deep = int(round(total_sleep * rng.uniform(0.14, 0.21)))
        rem = int(round(total_sleep * rng.uniform(0.19, 0.26)))
        # light is the exact remainder so the stages sum to total_sleep_time
        light = total_sleep - deep - rem
        rows.append(
            {
                "effective_time_frame_time_interval_start_date_time": _utc_str(pd.Series([bedtime]))[0],
                "effective_time_frame_time_interval_start_date_time_local": _local_str(pd.Series([bedtime]))[0],
                "effective_time_frame_time_interval_end_date_time": _utc_str(pd.Series([wake]))[0],
                "effective_time_frame_time_interval_end_date_time_local": _local_str(pd.Series([wake]))[0],
                "is_main_sleep": True,
                "sleep_stage_summary_total_sleep_time_value": total_sleep,
                "sleep_stage_summary_total_sleep_time_unit": "sec",
                "sleep_stage_summary_deep_sleep_duration_value": deep,
                "sleep_stage_summary_deep_sleep_duration_unit": "sec",
                "sleep_stage_summary_light_sleep_duration_value": light,
                "sleep_stage_summary_light_sleep_duration_unit": "sec",
                "sleep_stage_summary_rem_sleep_duration_value": rem,
                "sleep_stage_summary_rem_sleep_duration_unit": "sec",
                "sleep_stage_summary_sleep_efficiency_percentage_value": round(efficiency * 100, 2),
                "sleep_stage_summary_sleep_efficiency_percentage_unit": "%",
            }
        )
    df = pd.DataFrame(rows)
    starts = pd.to_datetime(df["effective_time_frame_time_interval_start_date_time"], utc=True)
    admin = _admin_columns("sleep_stage_summary", len(df), 30000, starts.reset_index(drop=True))
    for k, v in admin.items():
        df.insert(0, k, v)
    return df


# NOTE: there is deliberately no generator for `sleep_stage_episodes_*`.
#
# Study 30006's sleep-stage-summary records carry aggregate durations only --
# verified against the live Exchange: zero `sleep_stage_episodes_*` columns.
# Emitting them here would make `load_sleep_data(source="client")` appear to work
# in the demo while failing against every real study, which is the opposite of
# useful. Real stage episodes live in the Open mHealth sample instead, where
# `load_sleep_data(source="file")` can read them -- see demo-OMH.ipynb.


def build_heart_rate(start: pd.Timestamp, days: int, rng: np.random.Generator) -> pd.DataFrame:
    times = pd.date_range(start, periods=days * 96, freq="15min", tz=TZ_UTC)
    t_local = times.tz_convert(TZ_LOCAL)
    hours = t_local.hour + t_local.minute / 60.0
    hr = np.where((hours < 7) | (hours > 22), 58.0, 76.0) + rng.normal(0, 4.5, len(times))
    # an evening run on some days
    for day in sorted({d.date() for d in t_local}):
        if rng.random() < 0.45:
            s = pd.Timestamp(day, tz=TZ_LOCAL) + pd.Timedelta(hours=17, minutes=30)
            e = s + pd.Timedelta(minutes=40)
            m = np.asarray((t_local >= s) & (t_local < e))
            if m.any():
                hr[m] += rng.uniform(45, 70)
    hr = np.clip(hr, 45, 178)
    df = pd.DataFrame(_admin_columns("heart_rate", len(times), 60000, pd.Series(times)))
    df["heart_rate_value"] = np.round(hr, 0).astype(int)
    df["heart_rate_unit"] = "beats/min"
    df["effective_time_frame_time_interval_start_date_time"] = _utc_str(pd.Series(times)).values
    df["effective_time_frame_time_interval_end_date_time"] = _utc_str(
        pd.Series(times + pd.Timedelta(minutes=5))
    ).values
    return df


def build_oxygen_saturation(nights, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for bedtime, wake in nights:
        for t in pd.date_range(bedtime, wake, freq="30min"):
            rows.append(
                {
                    "effective_time_frame_date_time": _utc_str(pd.Series([t]))[0],
                    "effective_time_frame_date_time_local": _local_str(pd.Series([t]))[0],
                    "oxygen_saturation_value": round(float(np.clip(rng.normal(97.2, 1.1), 92, 100)), 1),
                    "oxygen_saturation_unit": "%",
                }
            )
    df = pd.DataFrame(rows)
    starts = pd.to_datetime(df["effective_time_frame_date_time"], utc=True).reset_index(drop=True)
    admin = _admin_columns("oxygen_saturation", len(df), 80000, starts)
    for k, v in admin.items():
        df.insert(0, k, v)
    return df


def build_food_entry(start: pd.Timestamp, days: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for d in range(days):
        day = start + pd.Timedelta(days=d)
        for hour, minute, name, carbs, kcal in MEALS:
            # local meal time -> UTC
            t = (pd.Timestamp(day.date(), tz=TZ_LOCAL) + pd.Timedelta(hours=hour, minutes=minute)).tz_convert(
                TZ_UTC
            )
            rows.append(
                {
                    "effective_time_frame_date_time": _utc_str(pd.Series([t]))[0],
                    "effective_time_frame_date_time_local": _local_str(pd.Series([t]))[0],
                    "food_name": name,
                    "carbohydrate_value": carbs,
                    "carbohydrate_unit": "g",
                    "calories_value": kcal,
                    "calories_unit": "kcal",
                    "meal_type": "breakfast" if hour < 11 else ("lunch" if hour < 16 else "dinner"),
                }
            )
    df = pd.DataFrame(rows)
    starts = pd.to_datetime(df["effective_time_frame_date_time"], utc=True).reset_index(drop=True)
    admin = _admin_columns("food_entry", len(df), 120000, starts)
    for k, v in admin.items():
        df.insert(0, k, v)
    return df


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="sample_data/jhe")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--start-date", default="2026-09-10",
                    help="first LOCAL calendar date covered by the record")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Local midnight, expressed in UTC -- so `start + h hours` is local h:00.
    start_local_midnight = pd.Timestamp(args.start_date, tz=TZ_LOCAL)
    start = start_local_midnight.tz_convert(TZ_UTC)
    days = args.days
    local_days = [(start_local_midnight + pd.Timedelta(days=d)).date() for d in range(days)]
    nights = night_windows(local_days, rng)

    builders = {
        "blood_glucose": lambda: build_blood_glucose(start, days, rng),
        "sleep_stage_summary": lambda: build_sleep_stage_summary(nights, rng),
        "heart_rate": lambda: build_heart_rate(start, days, rng),
        "oxygen_saturation": lambda: build_oxygen_saturation(nights, rng),
        "food_entry": lambda: build_food_entry(start, days, rng),
    }

    print(f"writing synthetic JHE sample data to {out}/")
    total = 0
    for name, fn in builders.items():
        df = fn()
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        total += len(df)
        print(f"  {path.name:32} {len(df):6} rows  {len(df.columns):2} cols  {path.stat().st_size/1024:8.1f} KB")
    print(f"  {'TOTAL':32} {total:6} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
