"""Forecasters.

Every forecaster has the signature ``f(ctx, horizon) -> np.ndarray`` where ``ctx`` is a
:class:`Context` holding **only information available at the forecast origin**:

* ``ctx.y``     - the target series up to (and including) the origin,
* ``ctx.panel`` - the VM-level panel truncated at the origin (or ``None``).

The evaluation harness (``cwfc.evaluate``) builds a fresh, truncated ``Context`` for every
origin, so a forecaster *cannot* read future values even by mistake.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .data import Panel

try:  # scikit-learn is only needed for the panel model
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception:  # pragma: no cover
    HistGradientBoostingRegressor = None


@dataclass
class Context:
    y: np.ndarray
    panel: Panel | None = None
    target: str = "aggregate_cpu_utilization"


Forecaster = Callable[[Context, int], np.ndarray]


# --------------------------------------------------------------------------- statistical
def persistence(ctx: Context, horizon: int) -> np.ndarray:
    """Last observed value."""
    return np.full(horizon, float(ctx.y[-1]))


def history_mean(ctx: Context, horizon: int) -> np.ndarray:
    """Mean of the whole available history."""
    return np.full(horizon, float(np.mean(ctx.y)))


def moving_average(k: int) -> Forecaster:
    def f(ctx: Context, horizon: int) -> np.ndarray:
        return np.full(horizon, float(np.mean(ctx.y[-k:])))

    f.__name__ = f"moving_average_{k}"
    return f


def seasonal_naive(season: int) -> Forecaster:
    """Repeat the last ``season`` observations (hourly season = 12 steps by default)."""

    def f(ctx: Context, horizon: int) -> np.ndarray:
        y = ctx.y
        if len(y) < season:
            return persistence(ctx, horizon)
        return np.array([y[-season + (i % season)] for i in range(horizon)], dtype=float)

    f.__name__ = f"seasonal_naive_{season}"
    return f


def ses(ctx: Context, horizon: int) -> np.ndarray:
    """Simple exponential smoothing; alpha chosen on a grid by in-sample one-step SSE."""
    y = np.asarray(ctx.y, dtype=float)
    best_sse, best_level = np.inf, float(y[-1])
    for a in np.linspace(0.05, 1.0, 20):
        level, sse = y[0], 0.0
        for v in y[1:]:
            sse += (v - level) ** 2
            level = a * v + (1 - a) * level
        if sse < best_sse:
            best_sse, best_level = sse, float(level)
    return np.full(horizon, best_level)


def ar1_around_mean(ctx: Context, horizon: int) -> np.ndarray:
    """AR(1) around the history mean (phi by OLS on the centred series, clipped to [0, 0.98])."""
    y = np.asarray(ctx.y, dtype=float)
    mu = y.mean()
    z = y - mu
    denom = float((z[:-1] ** 2).sum())
    phi = float(np.clip((z[1:] * z[:-1]).sum() / denom, 0.0, 0.98)) if denom > 1e-12 else 0.0
    return mu + phi ** np.arange(1, horizon + 1) * z[-1]


# --------------------------------------------------------------------------- panel helpers
def _filled_avg(panel: Panel) -> np.ndarray:
    """Panel ``avg`` with NaNs filled by the VM's own mean (fleet mean if the VM has no data)."""
    a = panel.avg
    if np.isfinite(a).all():
        return a
    with np.errstate(all="ignore"):
        vm_mean = np.nanmean(a, axis=1)
    fleet = np.nanmean(a) if np.isfinite(a).any() else 0.0
    vm_mean = np.where(np.isfinite(vm_mean), vm_mean, fleet)
    return np.where(np.isfinite(a), a, vm_mean[:, None])


def bottom_up_ar1(shrink_k: float = 30.0) -> Forecaster:
    """Per-VM AR(1) around each VM's history mean; phi_i shrunk to the fleet median; summed over VMs."""

    def f(ctx: Context, horizon: int) -> np.ndarray:
        x = _filled_avg(ctx.panel)
        n = x.shape[1]
        mu = x.mean(axis=1)
        z = x - mu[:, None]
        phi_i = np.clip((z[:, 1:] * z[:, :-1]).sum(1) / ((z[:, :-1] ** 2).sum(1) + 1e-9), 0, 0.98)
        w = n / (n + shrink_k)
        phi = w * phi_i + (1 - w) * np.median(phi_i)
        hh = np.arange(1, horizon + 1)[None, :]
        pred = mu[:, None] + (phi[:, None] ** hh) * (x[:, -1] - mu)[:, None]
        return np.clip(pred, 0, 100).mean(axis=0)

    f.__name__ = "bottom_up_ar1"
    return f


# --------------------------------------------------------------------------- panel GBM
FEATURE_NAMES = (
    "last", "lag2", "lag3", "mean3", "mean6", "mean12", "hist_mean", "hist_std", "hist_acf1",
    "max_last", "min_last", "max_mean6", "last_minus_mean", "mean6_minus_mean",
    "category", "cores", "memory_gb", "horizon",
)
TRAIN_HORIZONS = (1, 2, 4, 6, 9, 12, 16, 20, 24)  # horizons modelled directly; others interpolated
LOOKBACK = 12


def _vm_features(avg: np.ndarray, mx: np.ndarray, mn: np.ndarray, static: np.ndarray, s: int):
    """VM-level features computed from columns 0..s only (inclusive)."""
    x = avg[:, : s + 1]
    n = x.shape[1]
    last = x[:, -1]
    lag2 = x[:, -2] if n > 1 else last
    lag3 = x[:, -3] if n > 2 else last
    m3, m6, m12 = x[:, -3:].mean(1), x[:, -6:].mean(1), x[:, -12:].mean(1)
    hist_mean, hist_std = x.mean(1), x.std(1)
    z = x - hist_mean[:, None]
    acf1 = (z[:, 1:] * z[:, :-1]).sum(1) / ((z ** 2).sum(1) + 1e-9)
    mxl, mnl = mx[:, s], mn[:, s]
    mx6 = mx[:, max(0, s - 5) : s + 1].mean(1)
    feats = np.column_stack(
        [last, lag2, lag3, m3, m6, m12, hist_mean, hist_std, acf1, mxl, mnl, mx6, last - hist_mean, m6 - hist_mean, static]
    )
    return feats, hist_mean


def _training_rows(avg, mx, mn, static, rows, weights, stride):
    """Training rows from anchors s whose targets s+h lie strictly inside the history."""
    n = avg.shape[1]
    xs, ys, ws = [], [], []
    for s in range(LOOKBACK - 1, n - 1, stride):
        feats, hist_mean = _vm_features(avg, mx, mn, static, s)
        for h in TRAIN_HORIZONS:
            if s + h <= n - 1:
                xs.append(np.column_stack([feats[rows], np.full(len(rows), h)]))
                ys.append(avg[rows, s + h] - hist_mean[rows])
                ws.append(weights)
    return np.vstack(xs), np.concatenate(ys), np.concatenate(ws)


def _stratified_rows(hist_mean: np.ndarray, n_other: int, hot_threshold: float, rng):
    """Keep every high-load VM; sample the rest with inverse-probability weights.

    A uniform random VM subsample made the fitted model (and the forecast) depend heavily on *which*
    few hundred heavy hitters were drawn (see the robustness analysis in the report). Keeping all
    high-load VMs removes that luck factor while staying fast.
    """
    n = len(hist_mean)
    hot = np.where(hist_mean >= hot_threshold)[0]
    rest = np.where(hist_mean < hot_threshold)[0]
    if len(rest) <= n_other:
        return np.arange(n), np.ones(n)
    pick = rng.choice(rest, size=n_other, replace=False)
    rows = np.concatenate([hot, pick])
    weights = np.concatenate([np.ones(len(hot)), np.full(len(pick), len(rest) / len(pick))])
    return rows, weights


def panel_gbm(
    n_other: int = 1700,
    hot_threshold: float = 15.0,
    seed: int = 0,
    stride: int = 3,
    max_depth: int = 4,
    max_iter: int = 150,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 200,
    l2_regularization: float = 1.0,
) -> Forecaster:
    """Global gradient-boosting model across VMs, aggregated bottom-up.

    A single model maps VM-level features at an anchor time ``s`` (own recent values, own history
    statistics, last interval max/min, static size/category) plus the horizon ``h`` to
    ``avg_cpu(s+h) - own_history_mean``. It is trained on all (anchor, horizon) pairs whose targets
    lie *inside the available history* (all high-load VMs + an inverse-probability-weighted sample of the
    rest), then applied at the forecast origin to every VM and averaged.
    No fleet-level (common) feature is used. Early stopping is disabled (it would use a random row
    split); the number of iterations is fixed.
    """

    def f(ctx: Context, horizon: int) -> np.ndarray:
        if HistGradientBoostingRegressor is None:  # pragma: no cover
            raise ImportError("scikit-learn is required for the panel model")
        p = ctx.panel
        avg = _filled_avg(p)
        mx = np.where(np.isfinite(p.max), p.max, avg)
        mn = np.where(np.isfinite(p.min), p.min, avg)
        n_vms = avg.shape[0]
        rng = np.random.default_rng(seed)
        rows, weights = _stratified_rows(avg.mean(axis=1), n_other, hot_threshold, rng)
        x_tr, y_tr, w_tr = _training_rows(avg, mx, mn, p.static, rows, weights, stride)
        model = HistGradientBoostingRegressor(
            max_depth=max_depth, max_iter=max_iter, learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
            random_state=seed, early_stopping=False,
        ).fit(x_tr, y_tr, sample_weight=w_tr)
        feats, hist_mean = _vm_features(avg, mx, mn, p.static, avg.shape[1] - 1)
        grid_h = [h for h in TRAIN_HORIZONS if h <= max(horizon, 1)]
        if grid_h[-1] != horizon and horizon not in TRAIN_HORIZONS:
            grid_h.append(horizon)
        fleet = [
            float(np.clip(hist_mean + model.predict(np.column_stack([feats, np.full(n_vms, h)])), 0, 100).mean())
            for h in grid_h
        ]
        return np.interp(np.arange(1, horizon + 1), grid_h, fleet)

    f.__name__ = "panel_gbm"
    return f


def average_of(*forecasters: Forecaster) -> Forecaster:
    """Equal-weight forecast combination."""

    def f(ctx: Context, horizon: int) -> np.ndarray:
        return np.mean([g(ctx, horizon) for g in forecasters], axis=0)

    f.__name__ = "average_of_" + "+".join(g.__name__ for g in forecasters)
    return f


# --------------------------------------------------------------------------- registry
def candidates(target: str, with_panel: bool = True) -> dict[str, Forecaster]:
    """Ordered candidate set (simplest first - ties in CV are broken by this order)."""
    c: dict[str, Forecaster] = {
        "persistence": persistence,
        "history_mean": history_mean,
        "moving_average_6": moving_average(6),
        "moving_average_12": moving_average(12),
        "seasonal_naive_12": seasonal_naive(12),
        "ses": ses,
        "ar1_around_mean": ar1_around_mean,
        "blend_ar1_ses": average_of(ar1_around_mean, ses),  # hedge: mean-reverting + level-tracking
    }
    if with_panel and target == "aggregate_cpu_utilization":
        gbm = panel_gbm()
        c["bottom_up_ar1"] = bottom_up_ar1()
        c["panel_gbm"] = gbm
        c["hybrid_gbm_ar1"] = average_of(gbm, ar1_around_mean)
    return c
