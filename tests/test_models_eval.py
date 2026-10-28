import numpy as np
import pytest

from cwfc.evaluate import cross_validate, select_model
from cwfc.models import Context, ar1_around_mean, history_mean, moving_average, persistence, seasonal_naive, ses
from cwfc.validation import rolling_origins


def ctx(y):
    return Context(np.asarray(y, dtype=float))


def test_basic_forecasters():
    y = np.arange(1.0, 31.0)
    assert np.allclose(persistence(ctx(y), 5), 30.0)
    assert np.allclose(history_mean(ctx(y), 5), 15.5)
    assert np.allclose(moving_average(4)(ctx(y), 3), 28.5)
    assert np.allclose(seasonal_naive(12)(ctx(y), 24)[:12], y[-12:])
    for f in (ses, ar1_around_mean):
        out = f(ctx(y), 24)
        assert out.shape == (24,) and np.isfinite(out).all()


def test_ar1_on_constant_series_is_exact():
    assert np.allclose(ar1_around_mean(ctx(np.full(50, 10000.0)), 24), 10000.0)


def test_rolling_origins_never_exceed_data():
    o = rolling_origins(110, 24, 60, 3)
    assert o == [60, 63, 66, 69, 72, 75, 78, 81, 84]
    assert max(o) + 24 <= 110
    with pytest.raises(ValueError):
        rolling_origins(110, 0, 60)


def test_cv_hands_forecasters_only_truncated_data():
    y = np.arange(110, dtype=float)
    seen = []

    def spy(c, h):
        seen.append(len(c.y))
        assert c.y[-1] == len(c.y) - 1  # nothing beyond the origin is visible
        return np.full(h, c.y[-1])

    cv = cross_validate({"spy": spy}, y, None, "t", horizon=24, min_train=60, step=3)
    assert seen == cv.origins
    assert cv.actual.shape == (len(cv.origins), 24)


def test_select_model_minimax_and_tiebreak():
    y = np.sin(np.arange(110) / 5.0) + 10
    cv = cross_validate({"persistence": persistence, "history_mean": history_mean, "twin": persistence}, y, None, "t", 24, 48, 3)
    best, table = select_model(cv, (48, 60), ["persistence", "history_mean", "twin"])
    assert best in {"persistence", "history_mean"} and best != "twin"  # tie goes to the earlier candidate
    assert "worst_case_regret" in table
