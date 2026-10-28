"""Rolling-origin (expanding window) evaluation and model selection."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Panel
from .metrics import mae, rmse, bias
from .models import Context, Forecaster
from .validation import rolling_origins


@dataclass
class CVResult:
    origins: list[int]
    horizon: int
    actual: np.ndarray  # (O, H)
    preds: dict[str, np.ndarray] = field(default_factory=dict)  # model -> (O, H)
    seconds: dict[str, float] = field(default_factory=dict)

    def subset(self, min_origin: int) -> "CVResult":
        keep = [i for i, o in enumerate(self.origins) if o >= min_origin]
        return CVResult([self.origins[i] for i in keep], self.horizon, self.actual[keep], {k: v[keep] for k, v in self.preds.items()}, self.seconds)


def cross_validate(
    forecasters: dict[str, Forecaster],
    y: np.ndarray,
    panel: Panel | None,
    target: str,
    horizon: int,
    min_train: int,
    step: int = 3,
    verbose: bool = False,
) -> CVResult:
    """Expanding-window evaluation. ``y``/``panel`` must already be the *training window only*.

    For each origin ``o`` a fresh :class:`Context` is built from ``y[:o]`` and ``panel.truncate(o)``;
    forecasters only ever see data before ``o``.
    """
    y = np.asarray(y, dtype=float)
    origins = rolling_origins(len(y), horizon, min_train, step)
    if not origins:
        raise ValueError("training window too short for the requested rolling-origin evaluation")
    actual = np.vstack([y[o : o + horizon] for o in origins])
    preds = {k: np.zeros((len(origins), horizon)) for k in forecasters}
    seconds = {k: 0.0 for k in forecasters}
    for i, o in enumerate(origins):
        ctx = Context(y[:o].copy(), panel.truncate(o) if panel is not None else None, target)
        for name, f in forecasters.items():
            t0 = time.time()
            preds[name][i] = f(ctx, horizon)
            seconds[name] += time.time() - t0
        if verbose:
            print(f"  [{target}] origin {o} done", flush=True)
    return CVResult(origins, horizon, actual, preds, seconds)


def score_table(cv: CVResult, baseline: str = "persistence") -> pd.DataFrame:
    """MAE / RMSE / bias per model, with relative improvement over ``baseline``."""
    rows = []
    base_mae = mae(cv.actual, cv.preds[baseline]) if baseline in cv.preds else np.nan
    for name, p in cv.preds.items():
        m = mae(cv.actual, p)
        rows.append(
            {"model": name, "mae": m, "rmse": rmse(cv.actual, p), "bias": bias(cv.actual, p),
             f"mae_vs_{baseline}": (m / base_mae) if base_mae else np.nan}
        )
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def select_model(cv: CVResult, min_trains: tuple[int, ...], order: list[str]) -> tuple[str, pd.DataFrame]:
    """Pre-specified selection rule: minimise the *worst-case relative MAE* across evaluation windows.

    For every ``min_train`` in ``min_trains`` the CV origins ``>= min_train`` are scored; a model's
    regret in that window is ``MAE / best MAE``. The model with the smallest maximum regret wins,
    ties are broken by position in ``order`` (simplest first). Uses training-window data only.
    """
    table = {}
    for mt in min_trains:
        sub = cv.subset(mt)
        if not sub.origins:
            continue
        table[f"mae@min_train={mt}"] = {k: mae(sub.actual, p) for k, p in sub.preds.items()}
    df = pd.DataFrame(table).loc[order]
    eps = 1e-12
    regret = (df + eps).div(df.min(axis=0) + eps, axis=1)
    df["worst_case_regret"] = regret.max(axis=1)
    best = df["worst_case_regret"].round(10).idxmin()  # idxmin returns first minimum -> respects ``order``
    return str(best), df
