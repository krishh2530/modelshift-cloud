import numpy as np

from modelshift.drift.prediction_drift import compute_prediction_drift


def _extract_pred_map(pd_):
    if not isinstance(pd_, dict):
        return {}
    for k in ("prediction_drift", "prediction_drift_results", "results"):
        if k in pd_ and isinstance(pd_[k], dict):
            return pd_[k]
    return pd_


def test_prediction_drift_low_when_identical():
    base_p = np.linspace(0.05, 0.95, 200)
    live_p = base_p.copy()

    pd_ = compute_prediction_drift(base_p, live_p)
    pmap = _extract_pred_map(pd_)

    ks = float(pmap.get("ks_statistic", pmap.get("ks", 0.0)))
    assert ks == 0.0


def test_prediction_drift_high_when_inverted():
    # IMPORTANT:
    # KS drift measures DISTRIBUTION shift.
    # Inverting a UNIFORM set keeps the same distribution (KS ~ 0),
    # so we use a SKEWED distribution where inversion truly changes the distribution.

    rng = np.random.default_rng(0)
    base_p = rng.beta(2, 6, 800)   # skewed toward 0
    drift_p = 1.0 - base_p         # flips -> skewed toward 1 (big distribution shift)

    pd_ = compute_prediction_drift(base_p, drift_p)
    pmap = _extract_pred_map(pd_)

    ks = float(pmap.get("ks_statistic", pmap.get("ks", 0.0)))
    assert 0.30 <= ks <= 1.0
def test_prediction_drift_low_when_values_are_same_distribution():
    base_p = np.linspace(0.05, 0.95, 200)
    drift_p = base_p[::-1]  # same values, just reordered

    pd_ = compute_prediction_drift(base_p, drift_p)
    pmap = _extract_pred_map(pd_)

    ks = float(pmap.get("ks_statistic", pmap.get("ks", 0.0)))
    assert ks < 0.05