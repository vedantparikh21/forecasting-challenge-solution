import numpy as np
import pandas as pd

from cwfc.data import audit_cpu_table, build_panel, build_targets, load_challenge, make_grid, read_raw
from conftest import write_dataset


def test_holdout_rows_are_dropped_by_default(dataset):
    panel, hold = load_challenge(dataset, holdout=24)
    assert panel.n_steps == 110 and len(hold) == 24
    assert hold[0] == panel.timestamps[-1] + 300  # holdout follows training window immediately
    full, _ = load_challenge(dataset, holdout=24, include_holdout=True)
    assert full.n_steps == 134


def test_targets_count_each_vm_once_and_mean(dataset):
    cpu, meta = read_raw(dataset)
    dup = pd.concat([cpu, cpu.iloc[:50]], ignore_index=True)  # exact duplicates must not double count
    grid = make_grid(dup["timestamp"])
    panel = build_panel(dup, meta, grid)
    t = build_targets(panel)
    assert (t["active_vm_count"] == 120).all()
    first = cpu[cpu.timestamp == grid[0]]
    assert np.isclose(t["aggregate_cpu_utilization"].iloc[0], first["avg_cpu"].mean())


def test_gaps_and_invalid_values_are_not_invented(tmp_path):
    cpu, meta = write_dataset(tmp_path / "d", n_vms=30, n_steps=40)
    cpu = cpu[cpu.timestamp != 300 * 7]  # a whole timestamp missing
    cpu.loc[cpu.index[5], "avg_cpu"] = np.nan
    cpu.loc[cpu.index[9], "avg_cpu"] = 250.0  # outside [0, 100]
    a = audit_cpu_table(cpu)
    assert a["grid_gaps"] == 1 and a["cpu_outside_0_100"] >= 1
    panel = build_panel(cpu, meta, make_grid(cpu["timestamp"]))
    t = build_targets(panel)
    assert t["active_vm_count"].iloc[7] == 0 and np.isnan(t["aggregate_cpu_utilization"].iloc[7])
    assert t["active_vm_count"].iloc[0] == 30 - 2 + 1 or t["active_vm_count"].iloc[0] in (29, 30)
    assert t["aggregate_cpu_utilization"].between(0, 100).sum() == t["aggregate_cpu_utilization"].notna().sum()
