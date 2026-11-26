"""Step 6 - Complementary stock-market test, run once, exactly as declared in STOCK_TEST_PLAN.md.

Question: do acquirers that the model ranks as riskier do worse on the stock market over the next 3 years,
once delisted acquirers (taken over, bankrupt) are added back with Bloomberg prices?

Sample   : all out-of-sample predictions (acquisitions 2014-2016 from the development walk-forward, 2017-2022 from
           the final walk-forward). Models and predictions are not changed.
Prices   : Yahoo for listed acquirers; Bloomberg total-return index for the others (name-checked in
           merge_bloomberg.py; here the series must also exist before the entry month, so a ticker re-used by a
           newer company is never matched to an old deal).
Outcome  : BHAR36 = buy-and-hold return of the acquirer minus that of the US market over the 36 months starting the
           month after the acquisition-year 10-K is filed. After the last price of a delisted stock, the proceeds
           are reinvested in the market. At least one month of returns. Winsorised at 1% / 99%.
Test     : OLS of BHAR36 on the GBM risk percentile (rank within the fiscal year among all scored deals, 0 to 1) with
           fiscal-year fixed effects; standard errors clustered by entry month. Hypothesis: negative coefficient.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import portfolio as pf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stock_test"
HOLD = 36


def bhar_with_delisting(r, mkt, entry, last_m):
    """r: monthly returns of the stock (PeriodIndex); proceeds go to the market after the stock's last return."""
    window = pd.period_range(entry, entry + HOLD - 1, freq="M")
    window = window[window <= last_m]
    rs = r.reindex(window)
    if rs.notna().sum() == 0:
        return np.nan, 0
    m = mkt.reindex(window)
    path = rs.fillna(m)                      # after delisting (or a rare gap) the money earns the market return
    return float((1 + path).prod() - (1 + m).prod()), int(rs.notna().sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-run (reproduction); logged in run_log.txt")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / ".run_lock"
    # Run 1 (outputs/stock_test_run1_bug) used Bloomberg series that repeated the last price after a delisting, so
    # delisted stocks earned 0% instead of the market, contrary to the declared treatment. This run fixes that data
    # bug only (merge_bloomberg.py); the specification is unchanged and both runs are reported.
    if lock.exists() and not args.force:
        raise SystemExit("Stock test already run once (declared in STOCK_TEST_PLAN.md); not re-running.")
    lock.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))

    s = pd.concat([pd.read_csv(ROOT / "outputs" / p / "predictions.csv").assign(walk=p) for p in ("dev", "final")],
                  ignore_index=True)
    s["risk_pct"] = s.groupby("fyear")["GBM"].transform(lambda x: (x.rank(method="average") - 1) / (len(x) - 1))
    s["entry"] = pd.to_datetime(s["filed"]).dt.to_period("M") + 1

    ff = pd.read_csv(ROOT / "data" / "prices" / "ff_monthly.csv", index_col=0)
    ff.index = pd.PeriodIndex(ff.index, freq="M")
    mkt = ff["mkt_rf"] + ff["rf"]
    last_m = ff.index[-1]
    yr = pf.monthly_returns(pd.read_csv(ROOT / "data" / "prices" / "monthly_adjclose.csv", index_col=0))
    yr.index = pd.PeriodIndex(yr.index, freq="M")
    bfile = ROOT / "data" / "bloomberg" / "bbg_monthly_returns.csv"
    if bfile.exists():
        br = pd.read_csv(bfile, index_col=0)
        br.index = pd.PeriodIndex(br.index, freq="M")
        br.columns = br.columns.astype(int)
    else:   # Bloomberg data are licensed and not in the repository: the test then runs on Yahoo prices only
        print("WARNING: no Bloomberg returns found -> Yahoo-only (survivorship-biased) sample")
        br = pd.DataFrame(index=pd.PeriodIndex([], freq="M"))
    first_px = {c: br[c].first_valid_index() - 1 for c in br.columns}   # month of the first price

    rows = []
    for _, e in s.iterrows():
        src, r = None, None
        if isinstance(e["ticker"], str) and e["ticker"] in yr.columns:
            src, r = "Yahoo", yr[e["ticker"]]
        elif pd.notna(e["cik"]) and int(e["cik"]) in br.columns and first_px[int(e["cik"])] < e["entry"]:
            src, r = "Bloomberg", br[int(e["cik"])]
        b, n = bhar_with_delisting(r, mkt, e["entry"], last_m) if r is not None else (np.nan, 0)
        rows.append(dict(source=src if n > 0 else None, bhar36_raw=b, months_traded=n))
    s = pd.concat([s, pd.DataFrame(rows)], axis=1)

    d = s[s["bhar36_raw"].notna()].copy()
    lo, hi = d["bhar36_raw"].quantile([0.01, 0.99])
    d["bhar36"] = d["bhar36_raw"].clip(lo, hi)
    d["entry_m"] = d["entry"].astype(str)
    d["delisted"] = d["months_traded"] < HOLD

    print(f"Out-of-sample deals: {len(s):,} | with a BHAR36: {len(d):,} ({len(d) / len(s):.0%}) "
          f"| Yahoo {int((d['source'] == 'Yahoo').sum()):,} + Bloomberg {int((d['source'] == 'Bloomberg').sum()):,}")
    print(f"Deals whose stock stopped trading within 36 months (proceeds reinvested in the market): "
          f"{int(d['delisted'].sum()):,}")
    kn = s["fail"].notna()
    print("Accounting failure rate: deals with a BHAR {:.3f} | without {:.3f}".format(
        s.loc[kn & s["bhar36_raw"].notna(), "fail"].mean(), s.loc[kn & s["bhar36_raw"].isna(), "fail"].mean()))

    # ---- the declared test (one specification)
    fit = smf.ols("bhar36 ~ risk_pct + C(fyear)", data=d).fit(cov_type="cluster",
                                                              cov_kwds={"groups": pd.factorize(d["entry_m"])[0]})
    b, se, p = fit.params["risk_pct"], fit.bse["risk_pct"], fit.pvalues["risk_pct"]
    ci = fit.conf_int().loc["risk_pct"].tolist()
    print(f"\n[Declared test] BHAR36 on GBM risk percentile, fiscal-year FE, SE clustered by entry month "
          f"({d['entry_m'].nunique()} clusters):")
    print(f"  coefficient {b:+.3f} (riskiest vs safest deal), SE {se:.3f}, 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], "
          f"two-sided p = {p:.4f}, N = {int(fit.nobs):,}")

    # ---- descriptive: terciles of risk, with and without the Bloomberg-added acquirers
    d["tercile"] = pd.cut(d["risk_pct"], [-0.01, 1 / 3, 2 / 3, 1.01], labels=["Low risk", "Middle", "High risk"])
    desc = {}
    for lab, g in [("All (Yahoo + Bloomberg)", d), ("Yahoo only (survivors)", d[d["source"] == "Yahoo"]),
                   ("Bloomberg-added only", d[d["source"] == "Bloomberg"])]:
        t = g.groupby("tercile", observed=False)["bhar36"].agg(["mean", "median", "count"])
        t["share_bhar_below_-20%"] = g.groupby("tercile", observed=False)["bhar36"].apply(lambda x: (x <= -0.2).mean())
        desc[lab] = t
        print(f"\n{lab}: mean BHAR36 by risk tercile")
        print(t.round(3).to_string())
        print(f"  High - Low mean difference: {t.loc['High risk', 'mean'] - t.loc['Low risk', 'mean']:+.3f}")

    # ---- save
    keep = ["cik", "name", "fyear", "filed", "walk", "ticker", "source", "GBM", "risk_pct", "fail", "impair_ratio",
            "months_traded", "bhar36_raw"]
    s[keep].to_csv(OUT / "stock_test_events.csv", index=False)      # no Bloomberg prices, only derived returns
    pd.DataFrame([dict(coef=b, se=se, ci_lo=ci[0], ci_hi=ci[1], p_two_sided=p, n=int(fit.nobs),
                       clusters=d["entry_m"].nunique(), winsor_lo=lo, winsor_hi=hi)]).to_csv(OUT / "declared_test.csv",
                                                                                             index=False)
    pd.concat(desc, names=["sample", "tercile"]).to_csv(OUT / "bhar_by_tercile.csv")
    with open(OUT / "run_log.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} stock test run (force={args.force}) | coef {b:+.4f} p {p:.4f} n {int(fit.nobs)}\n")
    print("\nSaved to", OUT)


if __name__ == "__main__":
    main()
