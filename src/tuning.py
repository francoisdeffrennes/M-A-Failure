"""Step 8 - Robustness: hyperparameters chosen by time-series cross-validation INSIDE each training window.

Added after the main results (disclosed as a robustness check). The main models used fixed settings chosen before
the test. Here, for each test year Y (2017-2022), the settings are chosen using only acquisitions the model is
allowed to see (fiscal years <= Y-3), with the same timing rule as the real predictions:
    validation years V = Y-5, Y-4, Y-3 ; for each V, fit on fiscal years <= V-3, score year V
    (folds whose training data would be empty are skipped)
The setting with the best pooled validation AUC is refitted on all of years <= Y-3 and scores year Y.
Nothing from year Y or later is used to choose the settings.
"""
import itertools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import model as md
from run_all import load

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tuning"
YEARS = range(2017, 2023)
GBM_GRID = [dict(max_depth=d, learning_rate=lr, max_iter=it, min_samples_leaf=leaf, l2_regularization=l2)
            for d, lr, it, leaf, l2 in itertools.product([2, 3, 4], [0.03, 0.1], [100, 300], [20, 50], [0.0, 1.0])]
LOGIT_GRID = [dict(C=c) for c in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]]


def predict(kind, params, tr, te):
    y = tr["fail"].values
    if kind == "GBM":
        cats = sorted(tr["sector"].dropna().unique())
        enc = lambda X: np.c_[X[md.NUMERIC].values, X["sector"].map({c: i for i, c in enumerate(cats)}).values]
        m = HistGradientBoostingClassifier(categorical_features=[len(md.NUMERIC)], random_state=0, **params)
        return m.fit(enc(tr), y).predict_proba(enc(te))[:, 1]
    prep = md.Prep().fit(tr, md.NUMERIC)
    dtr, dte = md.sector_dummies(tr, te)
    m = LogisticRegression(max_iter=3000, **params).fit(np.c_[prep.transform(tr), dtr], y)
    return m.predict_proba(np.c_[prep.transform(te), dte])[:, 1]


def choose(kind, grid, ev, Y):
    """Best setting by pooled validation AUC over validation years inside the training window of test year Y."""
    folds = []
    for V in (Y - 5, Y - 4, Y - 3):
        tr = ev[(ev["fyear"] <= V - md.LAG) & ev["fail"].notna()]
        va = ev[(ev["fyear"] == V) & ev["fail"].notna()]
        if len(tr) >= 100 and len(va):
            folds.append((tr, va))
    res = []
    for p in grid:
        ys, ps = [], []
        for tr, va in folds:
            ys.append(va["fail"].values)
            ps.append(predict(kind, p, tr, va))
        res.append(roc_auc_score(np.concatenate(ys), np.concatenate(ps)))
    i = int(np.argmax(res))
    return grid[i], res[i], len(folds)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ev = load()
    ev = ev[ev["fyear"] <= max(YEARS)]
    out, log = [], []
    for Y in YEARS:
        tr = ev[(ev["fyear"] <= Y - md.LAG) & ev["fail"].notna()]
        te = ev[ev["fyear"] == Y].copy()
        for kind, grid, col in [("GBM", GBM_GRID, "GBM tuned"), ("Logit", LOGIT_GRID, "Logit (all) tuned")]:
            p, val_auc, nf = choose(kind, grid, ev, Y)
            pr = predict(kind, p, tr, pd.concat([tr, te]))
            te[col] = pr[len(tr):]
            te[col + "_hi"] = np.quantile(pr[:len(tr)], 2 / 3)
            log.append(dict(test_year=Y, model=col, n_folds=nf, validation_auc=val_auc, **p))
        out.append(te[["cik", "fyear", "accn", col] + ["GBM tuned", "GBM tuned_hi", "Logit (all) tuned",
                                                       "Logit (all) tuned_hi"]].loc[:, lambda d: ~d.columns.duplicated()])
        print(f"{Y}: done ({time.time() - t0:.0f}s)", flush=True)
    tuned = pd.concat(out)
    log = pd.DataFrame(log)

    # compare with the locked final predictions (same events)
    base = pd.read_csv(ROOT / "outputs" / "final" / "predictions.csv")
    d = base.merge(tuned, on=["cik", "fyear", "accn"], how="left", validate="1:1")
    k = d[d["fail"].notna()]
    models = ["Literature", "Logit (all)", "Logit (all) tuned", "GBM", "GBM tuned"]
    rows = {}
    for m in models:
        auc, lo, hi = md.auc_ci(d, m)
        rows[m] = dict(AUC=auc, ci_lo=lo, ci_hi=hi)
    aucs = pd.DataFrame(rows).T
    diffs = pd.DataFrame([dict(model=a, versus=b, **dict(zip(["auc_diff", "ci_lo", "ci_hi"], md.auc_diff_ci(d, a, b))))
                          for a, b in [("GBM tuned", "GBM"), ("GBM tuned", "Literature"),
                                       ("Logit (all) tuned", "Logit (all)"), ("Logit (all) tuned", "Literature")]])
    by_year = pd.DataFrame({m: k.groupby("fyear").apply(lambda g: roc_auc_score(g["fail"], g[m])) for m in models})
    flag = {}
    for m in models:
        f = k[m] >= k[m + "_hi"]
        flag[m] = dict(share_flagged=f.mean(), fail_if_flagged=k.loc[f, "fail"].mean(),
                       fail_if_not=k.loc[~f, "fail"].mean(), failures_caught=k.loc[f, "fail"].sum() / k["fail"].sum(),
                       impairments_caught=k.loc[f, "impair_next3"].fillna(0).sum() / k["impair_next3"].fillna(0).sum())
    flag = pd.DataFrame(flag).T

    pd.set_option("display.width", 200)
    print("\nSettings chosen inside each training window:")
    print(log.round(3).to_string(index=False))
    print(f"\nFinal test 2017-2022, {len(k):,} deals with known outcome. AUC with 95% bootstrap intervals:")
    print(aucs.round(3).to_string())
    print("AUC differences with 95% bootstrap intervals:")
    print(diffs.round(3).to_string(index=False))
    print("AUC by year:")
    print(by_year.round(3).to_string())
    print("Flag rule (above the training top-third cut):")
    print(flag.round(3).to_string())
    log.to_csv(OUT / "chosen_settings.csv", index=False)
    aucs.to_csv(OUT / "auc_tuned_vs_fixed.csv")
    # all Table 1 columns (AUC, average precision, Brier, failure rate in riskiest / safest fifth)
    md.scores(d, ["Base rate"] + models).to_csv(OUT / "table1_scores.csv")
    diffs.to_csv(OUT / "auc_differences.csv", index=False)
    by_year.to_csv(OUT / "auc_by_year.csv")
    flag.to_csv(OUT / "flag_rule.csv")
    tuned.to_csv(OUT / "tuned_predictions.csv", index=False)


if __name__ == "__main__":
    main()
