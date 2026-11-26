"""Step 10 - Corrected (real-time) walk-forward, added after the final audit.

Why: the pre-registered design trains the model for fiscal year Y on all deals of years <= Y-3. The outcome of a deal
of year Y-3 is only complete when the company files its 10-K for fiscal Y+0 (three years later), and that filing can
come AFTER some year-Y deals are reported (different fiscal-year ends and filing dates). The audit found that, on
average, 4-10% of the training labels used for a test deal were not yet public on that deal's filing date.

Correction: every test deal is scored by a model trained only on deals whose outcome was RESOLVED (the t+3 10-K, or
the last 10-K for a company that stopped filing after an impairment) before the first day of the month in which the
test deal's own 10-K was filed. Deals of the same filing month share one model. Hyperparameters of the tuned models are
chosen per test year on validation years Y-5..Y-3 under the same rule (validation labels resolved before the first
filing of year Y; their training labels resolved before the first filing of the validation year).

Outputs (outputs/corrected/): predictions_dev.csv, predictions_final.csv, chosen_settings.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import model as md
import tuning
from run_all import load

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "corrected"
PERIODS = {"dev": range(2014, 2017), "final": range(2017, 2023)}
BASE = ["Base rate", "Literature", "Logit (all)", "GBM"]
DEFAULTS = {"Logit": {"C": 1.0}, "GBM": {k: v for k, v in md.GBM_PARAMS.items() if k != "random_state"}}


def resolved_dates(ev):
    """Date on which each deal's 3-year outcome became public."""
    panel = pd.read_parquet(ROOT / "data" / "panel_firmyears.parquet")
    filed = panel.set_index(["cik", "fyear"])["filed"]
    filed = filed[~filed.index.duplicated()]
    out = []
    for c, y in zip(ev["cik"], ev["fyear"]):
        dates = [filed.get((c, y + k)) for k in (1, 2, 3)]
        dates = [pd.Timestamp(d) for d in dates if d is not None and pd.notna(d)]
        # the t+3 10-K if it exists, else the last 10-K filed (company stopped filing after an impairment)
        out.append(pd.Timestamp(filed.get((c, y + 3))) if pd.notna(filed.get((c, y + 3), np.nan)) else
                   (max(dates) if dates else pd.NaT))
    return pd.to_datetime(out)


def train_set(ev, Y, cutoff):
    return ev[(ev["fyear"] <= Y - md.LAG) & ev["fail"].notna() & (ev["resolved"] < cutoff)]


def choose(kind, grid, ev, Y):
    """Tuning per test year, with every label resolved before the relevant decision date."""
    cut_Y = ev.loc[ev["fyear"] == Y, "filed_dt"].min()
    folds = []
    for V in (Y - 5, Y - 4, Y - 3):
        va = ev[(ev["fyear"] == V) & ev["fail"].notna() & (ev["resolved"] < cut_Y)]
        tr = train_set(ev, V, ev.loc[ev["fyear"] == V, "filed_dt"].min())
        if len(tr) >= 100 and len(va) >= 30 and va["fail"].nunique() == 2:   # a usable validation year
            folds.append((tr, va))
    if not folds:
        return DEFAULTS[kind], 0, np.nan
    from sklearn.metrics import roc_auc_score
    res = []
    for p in grid:
        y = np.concatenate([va["fail"].values for _, va in folds])
        pr = np.concatenate([tuning.predict(kind, p, tr, va) for tr, va in folds])
        res.append(roc_auc_score(y, pr))
    i = int(np.argmax(res))
    return grid[i], len(folds), res[i]


def score_period(ev, years, log):
    out = []
    for Y in years:
        te_all = ev[ev["fyear"] == Y].copy()
        settings = {}
        for kind, grid, col in [("Logit", tuning.LOGIT_GRID, "Logit (all) tuned"), ("GBM", tuning.GBM_GRID, "GBM tuned")]:
            p, nf, va = choose(kind, grid, ev, Y)
            settings[col] = (kind, p)
            log.append(dict(test_year=Y, model=col, n_folds=nf, validation_auc=va, **p))
        for month, te in te_all.groupby(te_all["filed_dt"].dt.to_period("M")):
            tr = train_set(ev, Y, month.start_time)
            te = te.copy()
            both = pd.concat([tr, te])
            for name in BASE:
                p = md.fit_predict(name, tr, both)
                te[name], te[name + "_lo"], te[name + "_hi"] = p[len(tr):], np.quantile(p[:len(tr)], 1 / 3), np.quantile(p[:len(tr)], 2 / 3)
            for col, (kind, params) in settings.items():
                p = tuning.predict(kind, params, tr, both)
                te[col], te[col + "_lo"], te[col + "_hi"] = p[len(tr):], np.quantile(p[:len(tr)], 1 / 3), np.quantile(p[:len(tr)], 2 / 3)
            te["n_train"] = len(tr)
            out.append(te)
        print(f"{Y}: scored {len(te_all)} deals in {te_all['filed_dt'].dt.to_period('M').nunique()} monthly batches", flush=True)
    return pd.concat(out).sort_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = load()
    ev = ev[ev["fyear"] <= 2022].copy()
    ev["filed_dt"] = pd.to_datetime(ev["filed"])
    ev["resolved"] = resolved_dates(ev)
    lab = ev[ev["fail"].notna()]
    print(f"Deals with a known outcome: {len(lab):,}; resolution date missing: {int(lab['resolved'].isna().sum())}")
    log = []
    for period, years in PERIODS.items():
        s = score_period(ev, years, log)
        s.drop(columns=["filed_dt"]).to_csv(OUT / f"predictions_{period}.csv", index=False)
    pd.DataFrame(log).to_csv(OUT / "chosen_settings.csv", index=False)
    print(pd.DataFrame(log).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
