"""Shared helpers: tiny synthetic datasets in the exact challenge file format."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def write_dataset(path: Path, n_vms=120, n_steps=134, seed=0, drop_rows=None, constant_fleet=True):
    """Write vm_cpu_readings.csv.gz / vm_metadata.csv.gz with a mean-reverting synthetic workload."""
    rng = np.random.default_rng(seed)
    ts = np.arange(n_steps) * 300
    base = rng.gamma(1.2, 6.0, n_vms)  # heterogeneous VM levels
    common = np.cumsum(rng.normal(0, 0.05, n_steps)) * 0.2
    rows = []
    for i in range(n_vms):
        x = np.clip(base[i] + common + rng.normal(0, 1.0 + 0.1 * base[i], n_steps), 0.01, 99.0)
        rows.append(pd.DataFrame({"timestamp": ts, "vm_id": f"vm{i:04d}", "min_cpu": x * 0.5, "max_cpu": np.minimum(x * 1.8, 100.0), "avg_cpu": x}))
    cpu = pd.concat(rows, ignore_index=True)
    if drop_rows is not None:
        cpu = cpu.drop(index=cpu.sample(frac=drop_rows, random_state=seed).index)
    meta = pd.DataFrame(
        {
            "vm_id": [f"vm{i:04d}" for i in range(n_vms)],
            "subscription_id": "s", "deployment_id": "d", "timestamp_vm_created": 0, "timestamp_vm_deleted": 10**7,
            "max_cpu": 99.0, "avg_cpu": 5.0, "p95_max_cpu": 50.0,
            "vm_category": rng.choice(["Interactive", "Delay-insensitive", "Unknown"], n_vms),
            "virtual_core_bucket": rng.choice(["2", "4", "8", ">24"], n_vms),
            "memory_gb_bucket": rng.choice(["2", "4", "8", ">64"], n_vms),
        }
    )
    path.mkdir(parents=True, exist_ok=True)
    cpu.to_csv(path / "vm_cpu_readings.csv.gz", index=False)
    meta.to_csv(path / "vm_metadata.csv.gz", index=False)
    return cpu, meta


@pytest.fixture
def dataset(tmp_path):
    d = tmp_path / "data"
    write_dataset(d)
    return d
