"""Step 8b - Robustness to label timing (added during the final audit).

The main design trains the model for fiscal year Y on deals of years <= Y-3. The outcome of a deal of year Y-3 needs
the company's 10-K for fiscal Y, which is filed at about the same time as the year-Y deals being scored, and for some
fiscal-year ends up to about a year LATER. So a few training labels may not yet have been public on the decision date.
This script removes that possibility: every model (including the tuned one, with validation years shifted too) is
trained only on deals of years <= Y-4, whose outcomes were all public before any year-Y deal was reported.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import model as md
import tuning
from run_all import load

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "robustness"
YEARS = range(2017, 2023)
LAG = 4


def choose_lag(ev, Y, lag):
    folds = []
    for V in (Y - lag - 2, Y - lag - 1, Y - lag):
        tr = ev[(ev["fyear"] <= V - lag) & ev["fail"].notna()]
        va = ev[(ev["fyear"] == V) & ev["fail"].notna()]
        if len(tr) >= 100 and len(va):
            folds.append((tr, va))
    if not folds:                       # no history to validate on (2017): the pre-registered default C = 1
        return {"C": 1.0}, 0
    best, best_auc = None, -1
    for p in tuning.LOGIT_GRID:
        y = np.concatenate([va["fail"].values for _, va in folds])
        pr = np.concatenate([tuning.predict("Logit", p, tr, va) for tr, va in folds])
        a = roc_auc_score(y, pr)
        if a > best_auc:
            best, best_auc = p, a
    return best, len(folds)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = load()
    ev = ev[ev["fyear"] <= max(YEARS)]
    md.LAG = LAG
    scored = md.walk_forward(ev, YEARS)
    tuned, chosen = [], []
    for Y in YEARS:
        tr = ev[(ev["fyear"] <= Y - LAG) & ev["fail"].notna()]
        te = ev[ev["fyear"] == Y]
        p, nf = choose_lag(ev, Y, LAG)
        chosen.append(dict(test_year=Y, folds=nf, n_train=len(tr), **p))
        tuned.append(pd.Series(tuning.predict("Logit", p, tr, te), index=te.index))
    scored["Logit (all) tuned"] = pd.concat(tuned).reindex(scored.index).values
    models = ["Literature", "Logit (all)", "GBM", "Logit (all) tuned"]
    aucs = pd.DataFrame({m: dict(zip(["AUC", "ci_lo", "ci_hi"], md.auc_ci(scored, m))) for m in models}).T
    diff = pd.DataFrame([dict(model=a, versus=b, **dict(zip(["auc_diff", "ci_lo", "ci_hi"], md.auc_diff_ci(scored, a, b))))
                         for a, b in [("Logit (all) tuned", "Literature"), ("GBM", "Literature")]])
    print(f"Training on deals <= Y-{LAG}: settings chosen {pd.DataFrame(chosen).to_dict('records')}")
    print(f"Test 2017-2022, {int(scored['fail'].notna().sum()):,} deals with known outcome. AUC (95% bootstrap):")
    print(aucs.round(3).to_string())
    print(diff.round(3).to_string(index=False))
    aucs.to_csv(OUT / "lag4_auc.csv")
    diff.to_csv(OUT / "lag4_auc_differences.csv", index=False)


if __name__ == "__main__":
    main()
