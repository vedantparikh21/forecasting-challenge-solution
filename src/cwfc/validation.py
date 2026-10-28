from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon: int = 288,
    min_train_size: int = 7 * 24 * 12,
    step: int | None = None,
) -> list[TimeSplit]:
    """Return expanding-window splits for evenly sampled time series.

    `horizon=288` is one day at 5-minute cadence.
    """
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive")
    ts = pd.to_datetime(frame[timestamp_col], utc=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted")
    n = len(frame)
    step = horizon if step is None else step
    out: list[TimeSplit] = []
    train_end_idx = min_train_size
    while train_end_idx + horizon <= n:
        train_end = ts.iloc[train_end_idx - 1]
        val_start = ts.iloc[train_end_idx]
        val_end = ts.iloc[train_end_idx + horizon - 1]
        out.append(TimeSplit(train_end, val_start, val_end))
        train_end_idx += step
    return out


def rolling_origins(n_obs: int, horizon: int, min_train: int, step: int = 1) -> list[int]:
    """Forecast origins (= number of history points) for an expanding-window evaluation.

    An origin ``o`` means: fit on observations ``[0, o)`` and forecast ``[o, o + horizon)``.
    The last validation window ends exactly at ``n_obs``; no window ever extends beyond the data
    handed in, so passing only the training window guarantees the holdout is never touched.
    """
    if horizon <= 0 or min_train <= 0 or step <= 0:
        raise ValueError("horizon, min_train and step must be positive")
    return list(range(min_train, n_obs - horizon + 1, step))
