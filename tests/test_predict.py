import numpy as np
import pandas as pd

from cwfc.predict import main, run
from conftest import write_dataset


def test_schema_and_ranges(dataset, tmp_path):
    out = tmp_path / "predictions.csv"
    main(["--data", str(dataset), "--output", str(out), "--no-panel"])
    df = pd.read_csv(out)
    assert list(df.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert len(df) == 24 and df.timestamp.is_unique and df.timestamp.is_monotonic_increasing
    assert np.isfinite(df.to_numpy(dtype=float)).all()
    assert (df.active_vm_count >= 0).all() and df.aggregate_cpu_utilization.between(0, 100).all()
    raw = pd.read_csv(dataset / "vm_cpu_readings.csv.gz")
    assert list(df.timestamp) == list(np.sort(raw.timestamp.unique())[-24:])
    assert (df.active_vm_count == 120).all()  # constant fleet -> exact


def test_tiny_smoke_datasets_do_not_crash(tmp_path):
    for n_vms, n_steps in [(5, 40), (3, 30), (20, 134)]:
        d = tmp_path / f"d{n_vms}_{n_steps}"
        write_dataset(d, n_vms=n_vms, n_steps=n_steps)
        out, _ = run(d, tmp_path / f"p{n_vms}.csv", verbose=False, min_trains=(60,), step=12)
        assert len(out) == 24 and np.isfinite(out.to_numpy(dtype=float)).all()


def test_missing_readings_do_not_crash(tmp_path):
    d = tmp_path / "gappy"; write_dataset(d, n_vms=60, n_steps=134, drop_rows=0.05)
    out, _ = run(d, tmp_path / "p.csv", verbose=False, use_panel=False)
    assert len(out) == 24 and np.isfinite(out.to_numpy(dtype=float)).all()
