"""Step 6c - Descriptive market checks quoted in the manuscript, recomputed from the saved event-level returns
(outputs/stock_test/stock_test_events.csv), so they can be reproduced without the licensed Bloomberg prices.

  * share of acquirers trailing the market by 20 points or more over 36 months, mean BHAR36
  * AUC of each model's (accounting-trained) predictions for the market-failure label (BHAR36 <= -20%)
    and the combined label, on the 2017-2022 deals and on 2014-2022
  * Table 3 re-derived from the event file: declared OLS and the exploratory median regression point estimate
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stock_test"


def main():
    ev = pd.read_csv(OUT / "stock_test_events.csv")
    preds = pd.concat([pd.read_csv(ROOT / "outputs" / p / "predictions.csv") for p in ("dev", "final")],
                      ignore_index=True)
    assert len(ev) == len(preds) and (ev["name"].fillna("") == preds["name"].fillna("")).all()
    d = preds.assign(bhar=ev["bhar36_raw"].values, risk_pct=ev["risk_pct"].values, source=ev["source"].values)
    tf = ROOT / "outputs" / "tuning" / "tuned_predictions.csv"
    models = ["Literature", "Logit (all)", "GBM"]
    if tf.exists():
        d = d.merge(pd.read_csv(tf)[["cik", "fyear", "accn", "Logit (all) tuned"]], on=["cik", "fyear", "accn"], how="left")
    p = d[d["bhar"].notna()].copy()
    lo, hi = p["bhar"].quantile([0.01, 0.99])
    p["bhar_w"] = p["bhar"].clip(lo, hi)
    print(f"Deals with BHAR36: {len(p):,} | trailing the market by >= 20 pts: {(p['bhar_w'] <= -0.2).mean():.3f} "
          f"| mean BHAR36 (winsorised): {p['bhar_w'].mean():+.3f} | median: {p['bhar_w'].median():+.3f}")

    d["fail_mkt"] = np.where(d["bhar"].notna(), (d["bhar"] <= -0.2).astype(float), np.nan)
    d["fail_any"] = np.where((d["fail"] == 1) | (d["fail_mkt"] == 1), 1.0,
                             np.where(d["fail"].notna() & d["fail_mkt"].notna(), 0.0, np.nan))
    rows = []
    for period, g in [("2017-2022", d[d["fyear"] >= 2017]), ("2014-2022", d)]:
        for label in ["fail", "fail_mkt", "fail_any"]:
            k = g[g[label].notna()]
            ms = [m for m in models + ["Logit (all) tuned"] if m in k.columns and k[m].notna().all()]
            rows.append(dict(period=period, label=label, n=len(k), rate=k[label].mean(),
                             **{m: roc_auc_score(k[label], k[m]) for m in ms}))
    auc = pd.DataFrame(rows)
    print("\nAUC by failure definition (models trained on the accounting label):")
    print(auc.round(3).to_string(index=False))

    fit = smf.ols("bhar36 ~ risk_pct + C(fyear)", data=p.assign(bhar36=p["bhar_w"])).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(pd.to_datetime(p["filed"]).dt.to_period("M"))[0]})
    med = smf.quantreg("bhar36 ~ risk_pct + C(fyear)", p.assign(bhar36=p["bhar_w"])).fit(q=0.5).params["risk_pct"]
    print(f"\nTable 3 from the event file: OLS coef {fit.params['risk_pct']:+.4f} (p {fit.pvalues['risk_pct']:.4f}); "
          f"median-regression coef {med:+.4f}")
    auc.to_csv(OUT / "market_label_auc.csv", index=False)


if __name__ == "__main__":
    main()
