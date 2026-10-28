"""Generate the tables and figures used in reports/final_report.md.

    python -m cwfc.analysis --data challenge_data --predictions predictions.csv --out reports

Order of operations (protocol):
  1. EDA + rolling-origin CV use the TRAINING WINDOW ONLY.
  2. The holdout is read exactly once, at the very end, to *report* the error of the already-frozen
     ``predictions.csv`` and of the baselines. Nothing in steps 1-3 depends on it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import audit_cpu_table, build_targets, fill_internal_gaps, load_challenge, read_raw
from .evaluate import cross_validate, score_table
from .metrics import bias, mae, rmse, smape
from .models import candidates

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "violet": "#6250d6", "grey": "#8a8984", "ink": "#0b0b0b"}
H = 24


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "figure.dpi": 130, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})
    return plt


def eda(panel, targets, out: Path) -> dict:
    plt = _plt()
    A = panel.avg
    y = targets["aggregate_cpu_utilization"].to_numpy()
    hrs = (panel.timestamps - panel.timestamps[0]) / 3600.0
    res: dict = {}

    # fleet series
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].plot(hrs, y, color=C["blue"], lw=1.6)
    ax[0].axhline(y.mean(), color=C["grey"], ls="--", lw=1)
    ax[0].set(title="Fleet mean CPU (training window)", xlabel="hours", ylabel="% CPU")
    ax[1].plot(hrs[1:], np.diff(y), color=C["orange"], lw=1.2)
    ax[1].set(title="One-step change (mean-reverting)", xlabel="hours", ylabel="pp")
    fig.tight_layout(); fig.savefig(out / "figures/fig1_fleet_series.png"); plt.close(fig)

    # distribution
    vm_mean = A.mean(axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].hist(np.log10(np.clip(A.ravel(), 1e-2, 100)), bins=60, color=C["blue"])
    ax[0].set(title="VM-interval avg CPU (log10 %)", xlabel="log10 % CPU", ylabel="count")
    srt = np.sort(vm_mean)[::-1]
    ax[1].plot(np.arange(1, len(srt) + 1) / len(srt) * 100, np.cumsum(srt) / srt.sum() * 100, color=C["orange"], lw=2)
    ax[1].set(title="Load concentration", xlabel="top x% of VMs", ylabel="% of fleet CPU")
    fig.tight_layout(); fig.savefig(out / "figures/fig2_distribution.png"); plt.close(fig)

    top1, top10 = srt[: len(srt) // 100].sum() / srt.sum(), srt[: len(srt) // 10].sum() / srt.sum()
    dA = np.diff(A, axis=1)
    # diversification: variance of fleet diff vs independent benchmark
    var_fleet = float(dA.mean(0).var()); var_indep = float(dA.var(axis=1).mean() / A.shape[0])
    ac = [float(pd.Series(y).autocorr(k)) for k in range(1, 13)]
    res.update(
        n_vms=int(A.shape[0]), n_steps=int(A.shape[1]), fleet_mean=float(y.mean()), fleet_std=float(y.std()),
        fleet_cv_pct=float(100 * y.std() / y.mean()), fleet_min=float(y.min()), fleet_max=float(y.max()),
        vm_median=float(np.median(A)), vm_p90=float(np.percentile(A, 90)), vm_p99=float(np.percentile(A, 99)),
        share_top1pct_vms=float(top1), share_top10pct_vms=float(top10),
        var_fleet_diff=var_fleet, var_if_independent=var_indep, correlation_inflation=var_fleet / var_indep,
        fleet_acf_1_12=ac, trend_pp_per_hour=float(np.polyfit(np.arange(len(y)), y, 1)[0] * 12),
        multiplex_mean_vm_peak=float(A.max(axis=1).mean()), multiplex_fleet_peak=float(A.mean(0).max()),
    )

    # segments
    names = {i: n for i, n in enumerate(panel.category_names)}
    seg_rows = []
    for col, label in ((0, "vm_category"), (1, "virtual_cores")):
        for v in np.unique(panel.static[:, col]):
            m = panel.static[:, col] == v
            s = A[m].mean(0)
            seg_rows.append({"segment": label, "value": names.get(int(v), v) if col == 0 else v, "n_vms": int(m.sum()),
                             "mean_cpu": float(A[m].mean()), "share_of_cpu": float(A[m].sum() / A.sum()),
                             "series_cv_pct": float(100 * s.std() / s.mean())})
    pd.DataFrame(seg_rows).to_csv(out / "tables/segments.csv", index=False)

    # segment series figure
    fig, ax = plt.subplots(figsize=(6, 3))
    cols = [C["blue"], C["orange"], C["aqua"]]
    for i, v in enumerate(np.unique(panel.static[:, 0])):
        s = A[panel.static[:, 0] == v].mean(0)
        ax.plot(hrs, s - s.mean(), color=cols[i], lw=1.4, label=names.get(int(v), str(v)))
    ax.set(title="Segment means, de-meaned", xlabel="hours", ylabel="pp vs own mean"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "figures/fig3_segments.png"); plt.close(fig)
    return res


def run_cv(panel, targets, out: Path, step=3, min_train=48):
    y = fill_internal_gaps(targets["aggregate_cpu_utilization"].to_numpy())
    cands = candidates("aggregate_cpu_utilization", with_panel=True)
    cv = cross_validate(cands, y, panel, "aggregate_cpu_utilization", H, min_train=min_train, step=step)
    tabs = {}
    for mt in (48, 60):
        t = score_table(cv.subset(mt)); t.insert(0, "min_train", mt); tabs[mt] = t
        t.to_csv(out / f"tables/cv_min_train_{mt}.csv", index=False)
    return cv, y, cands


def peak_analysis(cv, selected: str, out: Path) -> pd.DataFrame:
    """Error on high-load windows (top-quartile mean actual) vs the rest."""
    lvl = cv.actual.mean(axis=1)
    hi = lvl >= np.quantile(lvl, 0.75)
    rows = []
    for name in ("persistence", "history_mean", selected):
        p = cv.preds[name]
        rows.append({"model": name, "mae_all": mae(cv.actual, p), "mae_peak_windows": mae(cv.actual[hi], p[hi]),
                     "bias_peak_windows": bias(cv.actual[hi], p[hi]), "mae_other_windows": mae(cv.actual[~hi], p[~hi])})
    df = pd.DataFrame(rows); df.to_csv(out / "tables/peak_analysis.csv", index=False)
    return df


def horizon_error(cv, selected: str, out: Path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3))
    for name, col in (("persistence", C["grey"]), ("history_mean", C["blue"]), (selected, C["orange"])):
        e = np.abs(cv.preds[name] - cv.actual).mean(axis=0)
        ax.plot(np.arange(1, H + 1) * 5, e, lw=1.8, color=col, label=name, ls="--" if name == "history_mean" and selected != name else "-")
    ax.set(title="CV MAE by forecast horizon", xlabel="minutes ahead", ylabel="MAE (pp)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "figures/fig4_horizon_error.png"); plt.close(fig)


def uncertainty(cv, selected: str, level=0.90):
    """Empirical (split-conformal-style) residual quantiles per horizon from CV, training window only."""
    err = cv.actual - cv.preds[selected]
    a = (1 - level) / 2
    lo = np.quantile(err, a, axis=0); hi = np.quantile(err, 1 - a, axis=0)
    cover = float(np.mean((err >= lo) & (err <= hi)))
    return lo, hi, cover


def holdout_report(data_dir, predictions, cands, y_train, panel_train, selected, lo, hi, out: Path) -> dict:
    """Single post-freeze look at the holdout (reporting only)."""
    plt = _plt()
    full, hold_ts = load_challenge(data_dir, include_holdout=True)
    tg = build_targets(full)
    pred = pd.read_csv(predictions)
    assert list(pred.timestamp) == list(hold_ts)
    truth = tg.loc[hold_ts]
    res: dict = {"active_vm_count": {}, "aggregate_cpu_utilization": {}}
    y_true = truth["aggregate_cpu_utilization"].to_numpy()
    from .models import Context

    ctx = Context(y_train.copy(), panel_train, "aggregate_cpu_utilization")
    rows = []
    for name, f in cands.items():
        p = np.clip(f(ctx, H), 0, 100)
        rows.append({"model": name, "mae": mae(y_true, p), "rmse": rmse(y_true, p), "bias": bias(y_true, p)})
    tab = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    pers = tab.loc[tab.model == "persistence", "mae"].iloc[0]
    tab["mae_vs_persistence"] = tab["mae"] / pers
    tab.to_csv(out / "tables/holdout_all_models_informational.csv", index=False)
    sub = pred["aggregate_cpu_utilization"].to_numpy()
    base_p = np.full(H, y_train[-1])
    res["aggregate_cpu_utilization"] = {
        "selected": selected, "mae": mae(y_true, sub), "rmse": rmse(y_true, sub), "bias": bias(y_true, sub),
        "smape_pct": smape(y_true, sub), "persistence_mae": mae(y_true, base_p), "persistence_rmse": rmse(y_true, base_p),
        "mae_vs_persistence": mae(y_true, sub) / mae(y_true, base_p), "holdout_actual_mean": float(y_true.mean()),
        "holdout_actual_min": float(y_true.min()), "holdout_actual_max": float(y_true.max()), "train_last": float(y_train[-1]),
        "interval90_coverage": float(np.mean((y_true >= sub + lo) & (y_true <= sub + hi))),
        "mean_interval_width_pp": float(np.mean(hi - lo)),
        "max_actual_minus_forecast": float(np.max(y_true - sub)),
        "headroom_q90_violations": int(np.sum(y_true > sub + hi)),
    }
    yt_c = truth["active_vm_count"].to_numpy(); sc = pred["active_vm_count"].to_numpy()
    res["active_vm_count"] = {"mae": mae(yt_c, sc), "rmse": rmse(yt_c, sc), "actual_min": float(yt_c.min()), "actual_max": float(yt_c.max())}

    hrs = (np.arange(H) + 1) * 5
    fig, ax = plt.subplots(figsize=(7, 3.2))
    n_ctx = 36
    ctx_x = np.arange(-n_ctx + 1, 1) * 5
    ax.plot(ctx_x, y_train[-n_ctx:], color=C["grey"], lw=1.4, label="training (last 3 h)")
    ax.plot(hrs, y_true, color=C["ink"], lw=1.6, label="holdout actual (reporting only)")
    ax.plot(hrs, sub, color=C["orange"], lw=2, label=f"forecast ({selected})")
    ax.fill_between(hrs, sub + lo, sub + hi, color=C["orange"], alpha=0.18, label="90% empirical band (CV residuals)")
    ax.plot(hrs, base_p, color=C["blue"], lw=1.4, ls="--", label="persistence baseline")
    ax.set(title="Holdout: forecast vs actual", xlabel="minutes relative to forecast origin", ylabel="fleet mean CPU %"); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out / "figures/fig5_holdout.png"); plt.close(fig)
    return res


def capacity_policy(cv, selected: str, out: Path) -> pd.DataFrame:
    """Backtest headroom rule  provision = forecast + q_k(CV residual)  using leave-one-origin-out quantiles."""
    err = cv.actual - cv.preds[selected]
    rows = []
    n = len(err)
    for k in (0.5, 0.8, 0.9, 0.95, 0.99):
        under, over = [], []
        for i in range(n):  # quantile of residuals from the OTHER origins (no look-ahead beyond the sample's own history is claimed)
            q = np.quantile(np.delete(err, i, axis=0), k, axis=0)
            gap = (cv.preds[selected][i] + q) - cv.actual[i]  # >0 = spare capacity (pp of fleet CPU)
            under.append(np.mean(gap < 0)); over.append(np.mean(np.maximum(gap, 0)))
        rows.append({"quantile": k, "violation_rate": float(np.mean(under)), "mean_spare_pp": float(np.mean(over))})
    df = pd.DataFrame(rows); df.to_csv(out / "tables/capacity_policy_backtest.csv", index=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="challenge_data"); ap.add_argument("--predictions", default="predictions.csv")
    ap.add_argument("--out", default="reports"); ap.add_argument("--selection-json", default="reports/tables/model_selection.json")
    a = ap.parse_args(argv)
    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True); (out / "tables").mkdir(parents=True, exist_ok=True)
    cpu, _ = read_raw(a.data)
    results = {"audit": audit_cpu_table(cpu)}
    panel, _ = load_challenge(a.data)  # training window only
    targets = build_targets(panel)
    results["eda"] = eda(panel, targets, out)
    cv, y, cands = run_cv(panel, targets, out)
    selected = json.loads(Path(a.selection_json).read_text())["aggregate_cpu_utilization"]["selected"]
    results["peak"] = json.loads(peak_analysis(cv, selected, out).to_json(orient="records"))
    horizon_error(cv, selected, out)
    lo, hi, cover = uncertainty(cv.subset(48), selected)
    results["uncertainty"] = {"in_sample_cv_coverage_90": cover, "lo": lo.tolist(), "hi": hi.tolist()}
    results["capacity_backtest"] = json.loads(capacity_policy(cv.subset(48), selected, out).to_json(orient="records"))
    results["holdout"] = holdout_report(a.data, a.predictions, cands, y, panel, selected, lo, hi, out)
    (out / "tables/results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({k: results[k] for k in ("peak", "capacity_backtest", "holdout")}, indent=2, default=float))


if __name__ == "__main__":
    main()
