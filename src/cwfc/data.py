"""Data loading, auditing and target construction.

Design rules (see README, "Leakage protocol"):

* The 24 holdout timestamps are identified from the *timestamp column only*.
* Unless ``include_holdout=True`` is passed explicitly (reporting code only), the
  holdout rows are dropped from the raw table *before* any CPU value is pivoted
  into an array, so no downstream code can read them by accident.
* Nothing is imputed silently: missing VM readings stay ``NaN`` in the panel.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STEP_SECONDS = 300  # five-minute cadence
HOLDOUT_STEPS = 24  # final two hours
CPU_FILE = "vm_cpu_readings.csv.gz"
META_FILE = "vm_metadata.csv.gz"
CPU_COLUMNS = ("timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu")

# Static, creation-time attributes that are known at any forecast origin.
# NOTE: the metadata columns ``avg_cpu``, ``max_cpu``, ``p95_max_cpu`` are lifetime
# statistics over the full 30-day trace (they include the holdout period and beyond),
# and ``timestamp_vm_deleted`` is a future event. They are deliberately NOT used as features.
STATIC_FEATURES = ("vm_category", "virtual_core_bucket", "memory_gb_bucket")


def bucket_to_number(value) -> float:
    """Map a bucket label such as ``'8'`` or ``'>24'`` to a number (``'>X'`` -> 1.5*X)."""
    s = str(value).strip()
    try:
        if s.startswith(">"):
            return float(s[1:]) * 1.5
        return float(s)
    except ValueError:
        return float("nan")


@dataclass(frozen=True)
class Panel:
    """VM x time arrays on a regular five-minute grid (``NaN`` = no valid reading)."""

    timestamps: np.ndarray  # (T,) int64 seconds
    vm_ids: np.ndarray  # (N,)
    avg: np.ndarray  # (N, T)
    max: np.ndarray  # (N, T)
    min: np.ndarray  # (N, T)
    static: np.ndarray  # (N, 3): category code, virtual cores, memory GB
    category_names: tuple = ()

    @property
    def n_vms(self) -> int:
        return int(self.avg.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.avg.shape[1])

    def truncate(self, n: int) -> "Panel":
        """Return the panel restricted to the first ``n`` time steps (a view-copy)."""
        return Panel(
            self.timestamps[:n].copy(),
            self.vm_ids,
            self.avg[:, :n].copy(),
            self.max[:, :n].copy(),
            self.min[:, :n].copy(),
            self.static,
            self.category_names,
        )


def read_raw(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Read the CPU readings and (optional) metadata tables."""
    data_dir = Path(data_dir)
    cpu_path = data_dir / CPU_FILE
    if not cpu_path.exists():
        raise FileNotFoundError(f"{cpu_path} not found. Extract the data-v1 release into {data_dir}/ (see README).")
    cpu = pd.read_csv(cpu_path)
    missing = set(CPU_COLUMNS) - set(cpu.columns)
    if missing:
        raise ValueError(f"{CPU_FILE} is missing columns: {sorted(missing)}")
    meta_path = data_dir / META_FILE
    meta = pd.read_csv(meta_path) if meta_path.exists() else None
    return cpu, meta


def make_grid(timestamps: pd.Series | np.ndarray, step: int = STEP_SECONDS) -> np.ndarray:
    """Regular grid spanning the observed timestamps (gaps become explicit grid points)."""
    ts = np.unique(np.asarray(timestamps, dtype="int64"))
    if len(ts) == 0:
        raise ValueError("no timestamps in the CPU table")
    first, last = int(ts[0]), int(ts[-1])
    return np.arange(first, last + 1, step, dtype="int64")


def audit_cpu_table(cpu: pd.DataFrame, step: int = STEP_SECONDS) -> dict:
    """Structural quality checks (cadence, coverage, duplicates, invalid values)."""
    ts = np.sort(cpu["timestamp"].unique())
    diffs = np.diff(ts)
    per_vm = cpu.groupby("vm_id").size()
    vals = cpu[["min_cpu", "max_cpu", "avg_cpu"]]
    return {
        "rows": int(len(cpu)),
        "n_vms": int(cpu["vm_id"].nunique()),
        "n_timestamps": int(len(ts)),
        "first_timestamp": int(ts[0]),
        "last_timestamp": int(ts[-1]),
        "span_hours": float((ts[-1] - ts[0] + step) / 3600.0),
        "cadence_seconds_counts": {int(k): int(v) for k, v in pd.Series(diffs).value_counts().items()},
        "grid_gaps": int(((diffs // step) - 1).clip(min=0).sum()) if len(diffs) else 0,
        "duplicate_vm_timestamp_rows": int(cpu.duplicated(["vm_id", "timestamp"]).sum()),
        "null_cells": {c: int(cpu[c].isna().sum()) for c in CPU_COLUMNS},
        "non_finite_cpu_cells": int((~np.isfinite(vals.to_numpy(dtype=float))).sum()),
        "cpu_outside_0_100": int(((vals < 0) | (vals > 100)).to_numpy().sum()),
        "avg_outside_min_max": int(((cpu["avg_cpu"] < cpu["min_cpu"] - 1e-9) | (cpu["avg_cpu"] > cpu["max_cpu"] + 1e-9)).sum()),
        "observations_per_vm": {int(k): int(v) for k, v in per_vm.value_counts().items()},
    }


def build_panel(cpu: pd.DataFrame, meta: pd.DataFrame | None, grid: np.ndarray) -> Panel:
    """Pivot the long CPU table onto ``grid``.

    Treatment of irregularities (documented in the report):
    * duplicate (vm, timestamp) rows: the first row is kept (VM counted once per timestamp);
    * non-finite values, or values outside [0, 100]: treated as *invalid* -> ``NaN``;
    * timestamps of the grid with no readings stay ``NaN`` (nothing is invented).
    """
    cpu = cpu.drop_duplicates(["vm_id", "timestamp"], keep="first")
    vm_ids = np.sort(cpu["vm_id"].unique())
    vm_index = pd.Index(vm_ids)
    ti = pd.Index(grid)
    rows = vm_index.get_indexer(cpu["vm_id"])
    cols = ti.get_indexer(cpu["timestamp"])
    keep = (cols >= 0) & (rows >= 0)
    arrays = {}
    for name in ("avg_cpu", "max_cpu", "min_cpu"):
        arr = np.full((len(vm_ids), len(grid)), np.nan)
        v = cpu[name].to_numpy(dtype=float)
        v = np.where(np.isfinite(v) & (v >= 0) & (v <= 100), v, np.nan)
        arr[rows[keep], cols[keep]] = v[keep]
        arrays[name] = arr
    # avg is the target: a reading is valid only if avg itself is valid
    invalid = ~np.isfinite(arrays["avg_cpu"])
    arrays["max_cpu"][invalid] = np.nan
    arrays["min_cpu"][invalid] = np.nan

    static = np.zeros((len(vm_ids), 3))
    categories: tuple = ()
    if meta is not None and "vm_id" in meta.columns:
        m = meta.drop_duplicates("vm_id").set_index("vm_id").reindex(vm_ids)
        if "vm_category" in m.columns:
            cat = m["vm_category"].astype("string").fillna("NA")
            categories = tuple(sorted(cat.unique()))
            static[:, 0] = pd.Categorical(cat, categories=categories).codes
        if "virtual_core_bucket" in m.columns:
            static[:, 1] = np.nan_to_num(m["virtual_core_bucket"].map(bucket_to_number).to_numpy(dtype=float), nan=0.0)
        if "memory_gb_bucket" in m.columns:
            static[:, 2] = np.nan_to_num(m["memory_gb_bucket"].map(bucket_to_number).to_numpy(dtype=float), nan=0.0)
    return Panel(np.asarray(grid, dtype="int64"), vm_ids, arrays["avg_cpu"], arrays["max_cpu"], arrays["min_cpu"], static, categories)


def load_challenge(data_dir: str | Path, holdout: int = HOLDOUT_STEPS, include_holdout: bool = False):
    """Load the challenge data.

    Returns ``(panel, holdout_timestamps)``. With the default ``include_holdout=False`` the
    panel contains only the training window (all but the final ``holdout`` grid steps) and the
    holdout rows have been dropped from the raw table before any value is read.
    """
    cpu, meta = read_raw(data_dir)
    grid = make_grid(cpu["timestamp"])
    if len(grid) <= holdout:
        raise ValueError(f"need more than {holdout} timestamps, got {len(grid)}")
    train_grid, holdout_ts = grid[:-holdout], grid[-holdout:]
    use = grid if include_holdout else train_grid
    cpu_use = cpu[cpu["timestamp"].isin(use)]  # holdout rows physically removed here
    return build_panel(cpu_use, meta, use), holdout_ts


# ----------------------------------------------------------------------------- targets
TARGETS = ("active_vm_count", "aggregate_cpu_utilization")


def build_targets(panel: Panel) -> pd.DataFrame:
    """Aggregate targets on the panel grid.

    * ``active_vm_count``: distinct VMs with a valid CPU reading at t (each VM counted once).
    * ``aggregate_cpu_utilization``: mean of valid VM-level ``avg_cpu`` at t (percent, [0, 100]);
      ``NaN`` if no VM has a valid reading at t (never imputed here).
    * ``core_weighted_cpu_utilization`` (extra): utilization weighted by virtual cores.
    """
    valid = np.isfinite(panel.avg)
    active = valid.sum(axis=0)
    total = np.where(valid, panel.avg, 0.0).sum(axis=0)
    agg = np.where(active > 0, total / np.maximum(active, 1), np.nan)
    cores = np.where(panel.static[:, 1] > 0, panel.static[:, 1], 1.0)[:, None]
    wsum = np.where(valid, cores, 0.0).sum(axis=0)
    cw = np.where(wsum > 0, np.where(valid, panel.avg * cores, 0.0).sum(axis=0) / np.maximum(wsum, 1e-12), np.nan)
    return pd.DataFrame(
        {
            "timestamp": panel.timestamps,
            "active_vm_count": active.astype(float),
            "aggregate_cpu_utilization": agg,
            "core_weighted_cpu_utilization": cw,
        }
    ).set_index("timestamp")


def fill_internal_gaps(y: np.ndarray) -> np.ndarray:
    """Linear interpolation of *internal* NaNs (model input only; evaluation ignores them)."""
    y = np.asarray(y, dtype=float).copy()
    bad = ~np.isfinite(y)
    if bad.all():
        return np.zeros_like(y)
    if bad.any():
        idx = np.arange(len(y))
        y[bad] = np.interp(idx[bad], idx[~bad], y[~bad])
    return y
