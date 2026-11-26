"""Run the whole study.

    python src/run_all.py --period dev     # acquisitions of 2014-2016 scored (training from 2010)
    python src/run_all.py --period final   # acquisitions of 2017-2022 scored; locked after the first run

Data steps (run once, in order): extract_sec.py -> build_panel.py -> fetch_meta.py -> fetch_prices.py
"""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import model as md
import portfolio as pf
from features import FEATURES, build

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PERIODS = {"dev": range(2014, 2017), "final": range(2017, 2023)}
EXCLUDE_SECTORS = {"Financial"}          # banks, insurers, REITs: goodwill works differently


def code_hash():
    h = hashlib.sha256()
    for f in sorted((ROOT / "src").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def load():
    ev = pd.read_parquet(ROOT / "data" / "events.parquet")
    panel = pd.read_parquet(ROOT / "data" / "panel_firmyears.parquet")
    meta = pd.read_csv(ROOT / "data" / "sec" / "meta.csv")
    e = build(ev, panel, meta)
    e = e[~e["sector"].isin(EXCLUDE_SECTORS)].copy()
    e["fail"] = e["fail_acct"]
    e["ticker"] = e["tickers"].fillna("").str.split("|").str[0].replace("", np.nan)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=PERIODS, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = ROOT / "outputs" / args.period
    out.mkdir(parents=True, exist_ok=True)
    if args.period == "final":
        lock = out / ".run_lock"
        if lock.exists() and not args.force:
            raise SystemExit("Final test already run; re-running would turn the test set into a tuning set.")
        lock.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(out / "run_log.txt", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} final run | code hash {code_hash()} | force={args.force}\n")

    years = PERIODS[args.period]
    e = load()
    e = e[e["fyear"] <= max(years)]                      # never touch later acquisitions
    px = pd.read_csv(ROOT / "data" / "prices" / "monthly_adjclose.csv", index_col=0)
    ff = pd.read_csv(ROOT / "data" / "prices" / "ff_monthly.csv", index_col=0)
    ff.index = pd.PeriodIndex(ff.index, freq="M")
    rets = pf.monthly_returns(px)
    # market definition of failure: 3-year buy-and-hold return at least 20 points below the market
    e["bhar36"] = pf.bhar(e, rets, ff["mkt_rf"] + ff["rf"])
    e["fail_mkt"] = np.where(e["bhar36"].notna(), (e["bhar36"] <= -0.20).astype(float), np.nan)
    e["fail_any"] = np.where((e["fail_acct"] == 1) | (e["fail_mkt"] == 1), 1.0,
                             np.where(e["fail_acct"].notna() & e["fail_mkt"].notna(), 0.0, np.nan))
    print(f"Mode {args.period} | code hash {code_hash()} | events used (fyear <= {max(years)}): {len(e):,}")
    scored = md.walk_forward(e, years)
    known = scored["fail"].notna()
    print(f"Scored acquisitions {min(years)}-{max(years)}: {len(scored):,} | outcome known: {known.sum():,} "
          f"| failure rate {scored.loc[known, 'fail'].mean():.3f} | training sizes {scored.groupby('fyear')['n_train'].first().to_dict()}")

    # ---- prediction quality
    sc = md.scores(scored)
    print("\n[Prediction] out-of-sample (higher AUC / avg precision better; lower Brier better):")
    print(sc.round(3).to_string())
    pairs = [("Literature", "Base rate"), ("Logit (all)", "Literature"), ("GBM", "Literature"), ("GBM", "Logit (all)")]
    ci = pd.DataFrame([dict(model=a, versus=b, **dict(zip(["auc_diff", "ci_lo", "ci_hi"], md.auc_diff_ci(scored, a, b))))
                       for a, b in pairs])
    aucs = pd.DataFrame({m: dict(zip(["AUC", "ci_lo", "ci_hi"], md.auc_ci(scored, m))) for m in md.MODELS[1:]}).T
    print("AUC with 95% bootstrap intervals (an interval above 0.5 = better than chance):")
    print(aucs.round(3).to_string())
    print("AUC differences with 95% bootstrap intervals:")
    print(ci.round(3).to_string(index=False))

    # robustness: unknown outcomes (company stopped filing without an impairment) counted as successes
    rob = scored.assign(fail=scored["fail"].fillna(0))
    print("\nRobustness - unknown outcomes treated as non-failures, AUC:",
          md.scores(rob)["AUC"].round(3).to_dict())
    rob2 = scored.assign(fail=np.where(scored["impair_ratio"] >= 0.25, 1.0, scored["fail"].where(scored["fail"] == 0)))
    print("Robustness - failure = impairment >= 25% of goodwill, AUC:", md.scores(rob2)["AUC"].round(3).to_dict())

    # share of all future impairment losses (scaled by the acquirer's size) falling in the riskiest fifth
    k = scored[known]
    loss = (k["impair_next3"] / k["assets_lag"]).clip(upper=1)
    cap = {m: loss[k[m] >= np.quantile(k[m], 0.8)].sum() / loss.sum() for m in md.MODELS[1:]}
    print("Share of impairment losses (as % of assets) in the riskiest 20% of deals (no skill = 0.20):",
          {m: round(v, 3) for m, v in cap.items()})

    # secondary definitions of failure (only acquirers with prices; survivorship-biased sample)
    sec = {}
    for label in ["fail_mkt", "fail_any"]:
        s2 = md.walk_forward(e.assign(fail=e[label]), years)
        sec[label] = md.scores(s2)
        print(f"\nSecondary label {label}: scored {s2['fail'].notna().sum():,}, failure rate {s2['fail'].mean():.3f}")
        print(sec[label].round(3).to_string())

    # ---- what drives the predictions (GBM fitted on all training data available for the last year)
    last = max(years)
    tr = e[(e["fyear"] <= last - md.LAG) & e["fail"].notna()]
    te = scored[known]
    cats = sorted(tr["sector"].dropna().unique())
    enc = lambda X: np.c_[X[md.NUMERIC].values, X["sector"].map({c: i for i, c in enumerate(cats)}).values]
    gbm = md.HistGradientBoostingClassifier(categorical_features=[len(md.NUMERIC)], **md.GBM_PARAMS).fit(enc(tr), tr["fail"])
    pi = permutation_importance(gbm, enc(te), te["fail"], scoring="roc_auc", n_repeats=20, random_state=0)
    imp = pd.Series(pi.importances_mean, index=md.NUMERIC + ["sector"]).sort_values(ascending=False)
    print("\n[Drivers] permutation importance (drop in AUC when the feature is shuffled):")
    print(imp.round(4).to_string())
    prep = md.Prep().fit(tr, md.NUMERIC)
    lr = md.LogisticRegression(C=1.0, max_iter=2000).fit(prep.transform(tr), tr["fail"])
    names = md.NUMERIC + [c + "_missing" for c in prep.flags]
    coef = pd.Series(lr.coef_[0], index=names).loc[md.NUMERIC].sort_values()
    print("Logit coefficients (standardised; positive = raises failure risk):")
    print(coef.round(3).to_string())
    q = scored[known].copy()
    q["gbm_quintile"] = pd.qcut(q["GBM"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    print("Failure rate by GBM risk quintile:", q.groupby("gbm_quintile")["fail"].mean().round(3).to_dict())

    # ---- economic value
    cov = scored["ticker"].isin(px.columns)
    print(f"\n[Economic value] acquirers with a Yahoo price series: {cov.sum():,} of {len(scored):,} ({cov.mean():.0%}); "
          f"failure rate with prices {scored.loc[cov & known, 'fail'].mean():.3f} vs without {scored.loc[~cov & known, 'fail'].mean():.3f}")
    res, series = {}, {}
    for mname in ["Literature", "Logit (all)", "GBM"]:
        h = pf.holdings(scored.assign(ticker=scored["ticker"].where(cov)), mname, mname + "_lo", mname + "_hi")
        months = pd.period_range(h["entry"].min(), min(h["exit"].max(), ff.index[-1]), freq="M")
        port = pf.portfolio_returns(h, rets, months)
        for col in port.columns:
            if mname != "GBM" and col in ("All acquirers",):
                continue
            key = col if col == "All acquirers" else f"{col} [{mname}]"
            series[key] = port[col]
            res[key] = pf.stats(port[col].rename(col), ff["rf"].reindex(months), ff["mkt_rf"].reindex(months))
    mkt = (ff["mkt_rf"] + ff["rf"]).reindex(months)
    res["Market (US)"] = pf.stats(mkt.rename("Market"), ff["rf"].reindex(months), ff["mkt_rf"].reindex(months))
    res["Market (US)"]["alpha_t"] = np.nan                # alpha of the market on itself is 0 by construction
    port_tab = pd.DataFrame(res).T
    print(f"Calendar-time portfolios {months[0]} -> {months[-1]}, hold {pf.HOLD} months, cost {pf.COST_BPS} bps per trade:")
    print(port_tab.round(3).to_string())

    # ---- save
    scored.to_csv(out / "predictions.csv", index=False)
    aucs.to_csv(out / "auc_intervals.csv")
    sc.to_csv(out / "prediction_scores.csv"); ci.to_csv(out / "auc_differences.csv", index=False)
    imp.to_csv(out / "permutation_importance.csv"); coef.to_csv(out / "logit_coefficients.csv")
    pd.concat(sec, names=["label", "model"]).to_csv(out / "secondary_labels.csv")
    pd.Series(cap).to_csv(out / "impairment_capture.csv")
    port_tab.to_csv(out / "portfolio_results.csv"); pd.DataFrame(series).to_csv(out / "portfolio_monthly_returns.csv")
    json.dump(dict(code_hash=code_hash(), period=args.period, years=list(years)), open(out / "manifest.json", "w"))
    print("\nSaved to", out)


if __name__ == "__main__":
    main()
