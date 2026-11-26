"""Checks of the rules the results depend on. Run: python tests/test_pipeline.py (after the pipeline has run).

1. No look-ahead: every prediction for fiscal year Y comes from a model trained on years <= Y-3 only.
2. Preprocessing is fitted on training data only (test rows cannot change the imputation or scaling).
3. Risk cut-offs come from training predictions (the alert threshold is known at the decision date).
4. BHAR with delisting: after the last return, the money earns the market return.
5. Tuning never looks at the test year.
6. Corrected (real-time) design: every training label was public before the scored deal's filing month.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model as md  # noqa: E402
from stock_test import bhar_with_delisting  # noqa: E402


def test_training_window():
    ev = pd.DataFrame({"fyear": np.repeat(np.arange(2010, 2020), 60)})
    rng = np.random.default_rng(0)
    for c in md.NUMERIC:
        ev[c] = rng.normal(size=len(ev))
    ev["sector"] = rng.choice(["A", "B", "C"], len(ev))
    ev["fail"] = (rng.random(len(ev)) < 0.3).astype(float)
    ev.loc[ev["fyear"] >= 2017, "fail"] = np.nan          # future outcomes unknown
    out = md.walk_forward(ev, [2016, 2017, 2018])
    for Y, g in out.groupby("fyear"):
        assert g["n_train"].iloc[0] == ((ev["fyear"] <= Y - 3) & ev["fail"].notna()).sum()
    print("ok  training window: model for year Y uses only years <= Y-3")


def test_prep_fitted_on_train_only():
    tr = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
    te = pd.DataFrame({"x": [1000.0, np.nan]})
    p = md.Prep().fit(tr, ["x"])
    z = p.transform(te)
    assert p.med["x"] == 2.0 and abs(z[1, 0] - (2.0 - p.mu["x"]) / p.sd["x"]) < 1e-12
    print("ok  preprocessing: medians and scaling come from training rows only")


def test_saved_predictions():
    for period in ("dev", "final"):
        f = ROOT / "outputs" / period / "predictions.csv"
        if not f.exists():
            print(f"skip saved predictions ({period}): run run_all.py first")
            continue
        s = pd.read_csv(f)
        assert s.groupby("fyear")["GBM_hi"].nunique().max() == 1, "one cut-off per year (from training)"
        print(f"ok  saved {period} predictions: one training-based cut-off per year, {len(s):,} deals")


def test_bhar_delisting():
    m = pd.Series(0.01, index=pd.period_range("2020-01", "2023-12", freq="M"))
    r = pd.Series([0.05] * 12, index=pd.period_range("2020-01", "2020-12", freq="M"))   # delisted after 12 months
    b, n = bhar_with_delisting(r, m, pd.Period("2020-01", "M"), m.index[-1])
    expected = (1.05 ** 12) * (1.01 ** 24) - 1.01 ** 36
    assert n == 12 and abs(b - expected) < 1e-12
    print("ok  BHAR: delisting proceeds reinvested in the market")


def test_tuning_no_test_year():
    import tuning
    seen = []
    orig = tuning.predict

    def spy(kind, params, tr, te):
        seen.append(tr["fyear"].max())
        return orig(kind, params, tr, te)

    tuning.predict = spy
    ev = pd.DataFrame({"fyear": np.repeat(np.arange(2010, 2020), 60)})
    rng = np.random.default_rng(1)
    for c in md.NUMERIC:
        ev[c] = rng.normal(size=len(ev))
    ev["sector"] = rng.choice(["A", "B"], len(ev))
    ev["fail"] = (rng.random(len(ev)) < 0.3).astype(float)
    tuning.choose("Logit", tuning.LOGIT_GRID[:2], ev, 2019)
    tuning.predict = orig
    assert max(seen) <= 2019 - 6, "validation models are trained on years <= V-3 <= Y-6"
    print("ok  tuning: settings chosen without the test year")


def test_realtime_labels_public():
    f = ROOT / "outputs" / "corrected" / "predictions_final.csv"
    if not f.exists():
        print("skip real-time check: run realtime.py first")
        return
    import realtime
    from run_all import load
    ev = load()
    ev = ev[ev["fyear"] <= 2022].copy()
    ev["filed_dt"] = pd.to_datetime(ev["filed"])
    ev["resolved"] = realtime.resolved_dates(ev)
    worst = pd.Timedelta(days=-100000)
    for Y in (2017, 2020, 2022):
        te = ev[ev["fyear"] == Y]
        for month in te["filed_dt"].dt.to_period("M").unique():
            tr = realtime.train_set(ev, Y, month.start_time)
            gap = tr["resolved"].max() - month.start_time
            assert gap < pd.Timedelta(0), "a training label resolved after the decision date"
            worst = max(worst, gap)
    print(f"ok  real-time design: latest training label resolved {-worst.days} days before the decision month")


if __name__ == "__main__":
    test_training_window()
    test_prep_fitted_on_train_only()
    test_saved_predictions()
    test_bhar_delisting()
    test_tuning_no_test_year()
    test_realtime_labels_public()
    print("all checks passed")
