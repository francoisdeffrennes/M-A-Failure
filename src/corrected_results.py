"""Step 11 - All results recomputed on the corrected (real-time) predictions of realtime.py.

Same metrics, alert rule, case-study rules and stock-market specifications as before; only the predictions change.
The stock tests reuse the event-level BHAR36 of stock_test.py (prices do not depend on the model) and replace the
risk percentile by the one from the corrected GBM predictions (the declared specification uses GBM).
Outputs: outputs/corrected/*.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

import model as md
from losses_avoided import table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "corrected"
MODELS = ["Base rate", "Literature", "Logit (all)", "GBM", "GBM tuned", "Logit (all) tuned"]


def main():
    pd.set_option("display.width", 220)
    f = pd.read_csv(OUT / "predictions_final.csv")
    dv = pd.read_csv(OUT / "predictions_dev.csv")
    k = f[f["fail"].notna()].copy()
    k["impair_next3"] = k["impair_next3"].fillna(0)
    print(f"Final test 2017-2022: {len(f):,} deals, {len(k):,} with known outcome, failure rate {k['fail'].mean():.3f}")
    print("Training-set sizes by year (min-max over monthly batches):",
          f.groupby("fyear")["n_train"].agg(["min", "max"]).to_dict("index"))

    # ---- Table 1
    sc = md.scores(f, MODELS)
    ci = pd.DataFrame({m: dict(zip(["AUC", "ci_lo", "ci_hi"], md.auc_ci(f, m))) for m in MODELS[1:]}).T
    t1 = sc.join(ci[["ci_lo", "ci_hi"]])
    pairs = [("Logit (all) tuned", "Literature"), ("Logit (all) tuned", "Logit (all)"), ("GBM", "Literature"),
             ("GBM tuned", "Literature"), ("Literature", "Base rate")]
    diff = pd.DataFrame([dict(model=a, versus=b, **dict(zip(["auc_diff", "ci_lo", "ci_hi"], md.auc_diff_ci(f, a, b))))
                         for a, b in pairs])
    by_year = pd.DataFrame({m: k.groupby("fyear").apply(lambda g: roc_auc_score(g["fail"], g[m])) for m in MODELS[1:]})
    print("\n[Table 1]\n", t1.round(3).to_string())
    print("\n[AUC differences]\n", diff.round(3).to_string(index=False))
    print("\n[AUC by year]\n", by_year.round(3).to_string())
    q = pd.qcut(k["Logit (all) tuned"].rank(method="first"), 5, labels=False)
    quint = k.groupby(q)["fail"].mean()
    print("\nFailure rate by quintile (tuned logit):", quint.round(3).tolist())
    rob1 = md.scores(f.assign(fail=f["fail"].fillna(0)), MODELS)["AUC"]
    rob2 = md.scores(f.assign(fail=np.where(f["impair_ratio"] >= 0.25, 1.0, f["fail"].where(f["fail"] == 0))), MODELS)["AUC"]
    print("Robustness AUC, unknown = success:", rob1.round(3).to_dict())
    print("Robustness AUC, threshold 25%:", rob2.round(3).to_dict())
    dev_auc = {m: round(roc_auc_score(dv.loc[dv['fail'].notna(), 'fail'], dv.loc[dv['fail'].notna(), m]), 3) for m in MODELS[1:]}
    print("Development years 2014-2016 AUC:", dev_auc)
    loss = (k["impair_next3"] / k["assets_lag"]).clip(upper=1)
    cap = {m: loss[k[m] >= np.quantile(k[m], 0.8)].sum() / loss.sum() for m in MODELS[1:]}
    print("Impairment losses (% of assets) in riskiest 20%:", {m: round(v, 3) for m, v in cap.items()})

    # ---- Table 2 and case studies
    t2 = table(k, ["No tool", "Literature", "GBM", "Logit (all) tuned"])
    print("\n[Table 2]\n", t2.T.round(3).to_string())
    st = pd.read_csv(ROOT / "outputs" / "stock_test" / "stock_test_events.csv")[["cik", "fyear", "filed", "bhar36_raw"]]
    c = k.merge(st.drop(columns="filed"), on=["cik", "fyear"], how="left", validate="1:1")
    cases = []
    for m in ["Logit (all) tuned", "GBM"]:
        c["risk_pct"] = c.groupby("fyear")[m].rank(pct=True)
        c["flag"] = c[m] >= c[m + "_hi"]
        for lab, sel, col in [("Hit", c["flag"] & (c["fail"] == 1), "impair_next3"),
                              ("Miss", ~c["flag"] & (c["fail"] == 1), "impair_next3"),
                              ("False alarm", c["flag"] & (c["fail"] == 0), "deal_value")]:
            r = c[sel].nlargest(3, col)
            for rank, (_, row) in enumerate(r.iterrows(), 1):
                cases.append(dict(model=m, case=lab, rank=rank, name=row["name"], fyear=row["fyear"],
                                  deal_value=row["deal_value"], probability=row[m], risk_pct=row["risk_pct"],
                                  impair_next3=row["impair_next3"], impair_ratio=row["impair_ratio"], bhar36=row["bhar36_raw"]))
    cs = pd.DataFrame(cases)
    print("\n[Case studies]\n", cs.round(3).to_string(index=False))

    # ---- stock-market tests with the corrected GBM percentile (declared specification)
    allp = pd.concat([dv, f], ignore_index=True)
    allp["risk_pct"] = allp.groupby("fyear")["GBM"].transform(lambda x: (x.rank(method="average") - 1) / (len(x) - 1))
    d = allp.merge(st, on=["cik", "fyear", "filed"], how="left", validate="1:1")
    d = d[d["bhar36_raw"].notna()].copy()
    lo, hi = d["bhar36_raw"].quantile([0.01, 0.99])
    d["bhar36"] = d["bhar36_raw"].clip(lo, hi)
    d["entry_m"] = (pd.to_datetime(d["filed"]).dt.to_period("M") + 1).astype(str)
    fit = smf.ols("bhar36 ~ risk_pct + C(fyear)", data=d).fit(cov_type="cluster",
                                                              cov_kwds={"groups": pd.factorize(d["entry_m"])[0]})
    b, ci_ = fit.params["risk_pct"], fit.conf_int().loc["risk_pct"].tolist()
    med = smf.quantreg("bhar36 ~ risk_pct + C(fyear)", d).fit(q=0.5).params["risk_pct"]
    rng = np.random.default_rng(0)
    groups = {g: x for g, x in d.groupby("entry_m")}
    keys = list(groups)
    draws = []
    for _ in range(1000):
        s = pd.concat([groups[g] for g in rng.choice(keys, len(keys))], ignore_index=True)
        try:
            draws.append(smf.quantreg("bhar36 ~ risk_pct + C(fyear)", s).fit(q=0.5, max_iter=2000).params["risk_pct"])
        except Exception:
            pass
    draws = np.array(draws)
    p_med = 2 * min((draws >= 0).mean(), (draws <= 0).mean())
    d["tercile"] = pd.cut(d["risk_pct"], [-0.01, 1 / 3, 2 / 3, 1.01], labels=["Low risk", "Middle", "High risk"])
    terc = d.groupby("tercile", observed=False)["bhar36"].agg(["mean", "median", "count"])
    t3 = pd.DataFrame([dict(test="OLS on mean, year FE (declared specification)", coef=b, ci_lo=ci_[0], ci_hi=ci_[1],
                            p=fit.pvalues["risk_pct"], n=int(fit.nobs)),
                       dict(test="Median regression, year FE (exploratory)", coef=med,
                            ci_lo=np.percentile(draws, 2.5), ci_hi=np.percentile(draws, 97.5), p=p_med, n=len(d))])
    print("\n[Table 3, corrected predictions]\n", t3.round(4).to_string(index=False))
    print("BHAR36 by GBM risk tercile:\n", terc.round(3).to_string())

    # market-failure label
    k2 = allp.merge(st, on=["cik", "fyear", "filed"], how="left")
    k2 = k2[(k2["fyear"] >= 2017) & k2["bhar36_raw"].notna()]
    mk = {m: round(roc_auc_score(k2["bhar36_raw"] <= -0.2, k2[m]), 3) for m in MODELS[1:]}
    print("Market-failure label AUC 2017-2022:", mk)

    # exploratory (not pre-specified): the years with enough history for tuning (2019-2022)
    late = f[f["fyear"] >= 2019]
    ex = md.auc_diff_ci(late, "Logit (all) tuned", "Literature")
    print(f"Exploratory, 2019-2022 only: tuned logit - literature AUC {ex[0]:+.3f} [{ex[1]:+.3f}, {ex[2]:+.3f}]")
    pd.Series(dict(zip(["auc_diff", "ci_lo", "ci_hi"], ex))).to_csv(OUT / "exploratory_2019_2022_diff.csv")

    # calendar-time portfolios (Yahoo prices, survivors only), as in run_all.py, on the corrected GBM groups
    import portfolio as pf
    px = pd.read_csv(ROOT / "data" / "prices" / "monthly_adjclose.csv", index_col=0)
    ff = pd.read_csv(ROOT / "data" / "prices" / "ff_monthly.csv", index_col=0)
    ff.index = pd.PeriodIndex(ff.index, freq="M")
    rets = pf.monthly_returns(px)
    ff_t = f.assign(ticker=f["tickers"].fillna("").str.split("|").str[0].replace("", np.nan))
    ff_t["ticker"] = ff_t["ticker"].where(ff_t["ticker"].isin(px.columns))
    port_res = {}
    for mname in ["GBM", "Logit (all) tuned"]:
        h = pf.holdings(ff_t, mname, mname + "_lo", mname + "_hi")
        months = pd.period_range(h["entry"].min(), min(h["exit"].max(), ff.index[-1]), freq="M")
        port = pf.portfolio_returns(h, rets, months)
        for col in port.columns:
            key = col if col == "All acquirers" else f"{col} [{mname}]"
            port_res[key] = pf.stats(port[col].rename(col), ff["rf"].reindex(months), ff["mkt_rf"].reindex(months))
    port_tab = pd.DataFrame(port_res).T
    print("\n[Portfolios, corrected groups]\n", port_tab.round(3).to_string())
    port_tab.to_csv(OUT / "portfolio_results.csv")

    t1.to_csv(OUT / "table1.csv"); diff.to_csv(OUT / "auc_differences.csv", index=False)
    by_year.to_csv(OUT / "auc_by_year.csv"); t2.to_csv(OUT / "table2_alert_rule.csv")
    cs.to_csv(OUT / "case_studies.csv", index=False); t3.to_csv(OUT / "table3_stock.csv", index=False)
    terc.to_csv(OUT / "bhar_by_tercile.csv")
    pd.Series(quint.values, name="failure_rate").to_csv(OUT / "quintiles.csv")
    pd.DataFrame({"unknown_as_success": rob1, "threshold_25pct": rob2}).to_csv(OUT / "robustness.csv")
    pd.Series(mk).to_csv(OUT / "market_label_auc.csv"); pd.Series(dev_auc).to_csv(OUT / "dev_auc.csv")
    pd.Series(cap).to_csv(OUT / "impairment_capture_top20.csv")


if __name__ == "__main__":
    main()
