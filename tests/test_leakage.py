"""The core protocol test: the 24 holdout observations must not influence predictions."""
import numpy as np
import pandas as pd

from cwfc.predict import run
from conftest import write_dataset

FAST = dict(min_trains=(60,), step=12)  # 3 rolling origins keeps the test quick; the protocol is identical


def _perturb_holdout(src, dst, factor):
    cpu = pd.read_csv(src / "vm_cpu_readings.csv.gz")
    ts = np.sort(cpu.timestamp.unique())
    hold = cpu.timestamp.isin(ts[-24:])
    for c in ("min_cpu", "max_cpu", "avg_cpu"):
        cpu.loc[hold, c] = np.minimum(cpu.loc[hold, c] * factor, 100.0)
    dst.mkdir(parents=True, exist_ok=True)
    cpu.to_csv(dst / "vm_cpu_readings.csv.gz", index=False)
    (dst / "vm_metadata.csv.gz").write_bytes((src / "vm_metadata.csv.gz").read_bytes())


def test_predictions_do_not_depend_on_holdout_values(tmp_path):
    a = tmp_path / "a"; write_dataset(a, n_vms=80, n_steps=134, seed=3)
    b = tmp_path / "b"; _perturb_holdout(a, b, factor=4.0)   # wildly different holdout
    out_a, _ = run(a, tmp_path / "pa.csv", verbose=False, **FAST)
    out_b, _ = run(b, tmp_path / "pb.csv", verbose=False, **FAST)
    pd.testing.assert_frame_equal(out_a, out_b, check_exact=True)


def test_panel_models_also_ignore_holdout(tmp_path):
    a = tmp_path / "a"; write_dataset(a, n_vms=80, n_steps=134, seed=4)
    b = tmp_path / "b"; _perturb_holdout(a, b, factor=0.1)
    out_a, sel = run(a, tmp_path / "pa.csv", verbose=False, use_panel=True, **FAST)
    out_b, _ = run(b, tmp_path / "pb.csv", verbose=False, use_panel=True, **FAST)
    assert "panel_gbm" in sel["aggregate_cpu_utilization"]["candidates"]
    pd.testing.assert_frame_equal(out_a, out_b, check_exact=True)
