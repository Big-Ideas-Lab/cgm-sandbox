"""Draw every overlay and extension, and fail loudly if any cannot render.

Why this exists
---------------
Overlay tests previously only *constructed* overlays. A stubbed exercise
skeleton -- ``draw()`` full of ``...`` bodies -- constructs perfectly happily and
only explodes at render time, so the breakage stayed invisible.

That is exactly what happened to ``MeanGlucoseOverlay``: it shipped as an
unfinished hands-on exercise with 7 TODOs and ellipsis bodies, while its working
implementation sat unused in ``solutions/mean_gl_overlay.py``. Construction-only
checks, and behaviour-preserving refactor checks, both pass over that kind of bug.

So this exercises the *drawing* path for every overlay, one at a time, so a
failure is attributed to a specific class rather than to the viewer.

Run:  python tools/check_overlays_draw.py
"""

from __future__ import annotations

import inspect
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

# Allow running straight from a fresh clone: `python tools/check_overlays_draw.py`
# puts tools/ (not the repo root) on sys.path, so add the repo root explicitly
# rather than requiring `pip install -e .` first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cgmsandbox import (  # noqa: E402
    CGMViewer,
    CgmMeasuresOverlay,
    CvOverlay,
    FoodEntryOverlay,
    HypnogramExtension,
    MageOverlay,
    MeanGlucoseOverlay,
    PPGROverlay,
    TimeInRangeOverlay,
    WakeupGlucoseOverlay,
)

BASE = "sample_data/omh"  # Open mHealth sample -- has real sleep stage episodes

# (label, callable(viewer) -> overlay-or-extension, is_extension)
CASES = [
    ("TimeInRangeOverlay", lambda: TimeInRangeOverlay(), False),
    ("FoodEntryOverlay", lambda: FoodEntryOverlay(
        source="file", base_path=BASE, filename="food_entry.json"), False),
    ("CgmMeasuresOverlay", lambda: CgmMeasuresOverlay(), False),
    ("CvOverlay", lambda: CvOverlay(), False),
    ("MageOverlay", lambda: MageOverlay(show_ma=True), False),
    ("WakeupGlucoseOverlay", lambda: WakeupGlucoseOverlay(
        source="file", base_path=BASE, filename="sleep_stage_summary.json"), False),
    ("PPGROverlay", lambda: PPGROverlay(
        source="file", base_path=BASE, filename="food_entry.json"), False),
    ("MeanGlucoseOverlay(scope='daily')", lambda: MeanGlucoseOverlay(
        source="file", scope="daily"), False),
    ("MeanGlucoseOverlay(scope='full')", lambda: MeanGlucoseOverlay(
        source="file", scope="full"), False),
    ("HypnogramExtension", lambda: HypnogramExtension(
        source="file", base_path=BASE, filename="sleep_stage_summary.json"), True),
]


def main() -> int:
    print("=" * 78)
    print("overlay / extension DRAW check")
    print("=" * 78)

    passed, failed = 0, []
    total = 0
    for label, factory, is_extension in CASES:
        # Extensions only get a dedicated panel in daily mode: _render_full
        # builds two week-panels and no extension axes. So full mode is only
        # exercised for overlays.
        modes = ["daily"] if is_extension else ["daily", "full"]
        for mode in modes:
            total += 1
            tag = f"{label} [{mode}]"
            try:
                # A fresh viewer per case so one failure cannot poison the next.
                viewer = CGMViewer(source="file", base_path=BASE,
                                   filename="blood_glucose.json", gl_range=(0, 250))
                target = factory()
                if is_extension:
                    viewer.add_extensions(target)
                else:
                    viewer.add_overlay(target)
                viewer.selected_date = viewer.unique_days[min(1, len(viewer.unique_days) - 1)]
                # render() is the entry point that sets viewer.view_mode, which
                # scale() reads. Calling _render_day() directly bypasses that and
                # raises a misleading AttributeError about view_mode.
                viewer.render(view_mode=mode)
                print(f"  [PASS] {tag}")
                passed += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {tag}  {type(exc).__name__}: {exc}")
                failed.append((tag, traceback.format_exc()))

    print("=" * 78)
    print(f"RESULT: {passed}/{total} draw checks passed")
    for label, tb in failed:
        print(f"\n--- {label} ---")
        print(tb)
    print("=" * 78)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
