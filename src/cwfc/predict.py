"""One-command pipeline: challenge data -> ``predictions.csv``.

    python -m cwfc.predict --data challenge_data --output predictions.csv

Steps (all data-driven; nothing is hard-coded to the public release):

1. Read the CPU table, identify the final ``--holdout`` grid timestamps from the timestamp column
   alone and **drop the holdout rows** before any value is read.
2. Build the two required targets on the five-minute grid from the training window.
3. For each target, evaluate the candidate forecasters with rolling-origin (expanding-window)
   validation *inside the training window* and select a model with the pre-specified
   minimax-regret rule (``cwfc.evaluate.select_model``).
4. Refit the selected model on the full training window and forecast the holdout timestamps.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import HOLDOUT_STEPS, TARGETS, build_targets, fill_internal_gaps, load_challenge
from .evaluate import cross_validate, select_model
from .models import Context, candidates

DEFAULT_MIN_TRAINS = (48, 60)  # evaluation windows start with >= 4 h / 5 h of history
FALLBACK = {"active_vm_count": "persistence", "aggregate_cpu_utilization": "history_mean"}


def forecast_target(panel, y, target, horizon, min_trains, step, use_panel, verbose=True):
    """Select (rolling-origin CV on the training window) and fit a model for one target."""
    n = len(y)
    with_panel = use_panel and panel.n_vms >= 50 and n >= 40
    cands = candidates(target, with_panel=with_panel)
    feasible = tuple(m for m in min_trains if n - horizon >= m)
    if not feasible and n - horizon >= 10:
        feasible = (n - horizon,)  # tiny data: a single evaluation origin
    info: dict = {"target": target, "n_train": int(n), "n_vms": int(panel.n_vms), "candidates": list(cands)}
    if not feasible:
        chosen = FALLBACK[target]
        info.update(selected=chosen, selection="fallback (training window too short for rolling-origin validation)")
    else:
        t0 = time.time()
        cv = cross_validate(cands, y, panel, target, horizon, min_train=min(feasible), step=step, verbose=False)
        chosen, table = select_model(cv, feasible, list(cands))
        info.update(selected=chosen, selection="minimax regret across evaluation windows", min_trains=list(feasible),
                    n_origins=len(cv.origins), cv_table=json.loads(table.to_json()), cv_seconds=round(time.time() - t0, 1))
        if verbose:
            print(f"[{target}] CV done in {time.time() - t0:.0f}s ->\n{table.round(4).to_string()}", flush=True)
    ctx = Context(y.copy(), panel.truncate(panel.n_steps), target)
    fc = np.asarray(cands[chosen](ctx, horizon), dtype=float)
    return fc, info


def run(data_dir, output, holdout=HOLDOUT_STEPS, min_trains=DEFAULT_MIN_TRAINS, step=3, use_panel=True,
        selection_json=None, verbose=True):
    t0 = time.time()
    panel, holdout_ts = load_challenge(data_dir, holdout=holdout, include_holdout=False)
    targets = build_targets(panel)
    out = pd.DataFrame({"timestamp": holdout_ts.astype("int64")})
    selections = {}
    for target in TARGETS:
        y = fill_internal_gaps(targets[target].to_numpy())
        fc, info = forecast_target(panel, y, target, holdout, tuple(min_trains), step, use_panel, verbose)
        if target == "active_vm_count":
            fc = np.maximum(fc, 0.0)
        else:
            fc = np.clip(fc, 0.0, 100.0)
        out[target] = fc
        selections[target] = info
        if verbose:
            print(f"[{target}] selected: {info['selected']}", flush=True)
    out = out[["timestamp", "active_vm_count", "aggregate_cpu_utilization"]]
    assert len(out) == holdout and np.isfinite(out.to_numpy(dtype=float)).all() and out["timestamp"].is_unique
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    if selection_json:
        Path(selection_json).parent.mkdir(parents=True, exist_ok=True)
        Path(selection_json).write_text(json.dumps(selections, indent=2))
    if verbose:
        print(f"wrote {output} ({len(out)} rows) in {time.time() - t0:.0f}s", flush=True)
    return out, selections


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate predictions.csv for the cloud workload forecasting challenge")
    ap.add_argument("--data", default="challenge_data", help="directory containing vm_cpu_readings.csv.gz (+ vm_metadata.csv.gz)")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--holdout", type=int, default=HOLDOUT_STEPS, help="number of final timestamps to forecast")
    ap.add_argument("--min-train", type=int, nargs="+", default=list(DEFAULT_MIN_TRAINS), help="evaluation windows (min history)")
    ap.add_argument("--step", type=int, default=3, help="spacing between rolling origins (5-min steps)")
    ap.add_argument("--no-panel", action="store_true", help="skip the VM-level (panel) models")
    ap.add_argument("--selection-json", default=None, help="optional path to dump the model-selection table")
    a = ap.parse_args(argv)
    run(a.data, a.output, a.holdout, tuple(a.min_train), a.step, not a.no_panel, a.selection_json)


if __name__ == "__main__":
    main()
