"""Step 7 - What would the tool have flagged for an acquirer? (complementary, after the main results)

Rule usable in real time: a deal is flagged "high risk" when its predicted probability is above the top-third cut
point of the model's predictions on its own training sample (known at the decision date, no look-ahead).
For the deals of the locked final test (acquisitions 2017-2022, outcome known), compare:
    no tool | Literature rule (4 classic variables) | GBM
on: share of deals flagged, failure rate of flagged vs not flagged, goodwill impairments ($bn) falling in flagged
deals, and good deals flagged by mistake (false alarms).
Caveat: impairments are firm-level; they may include goodwill from earlier deals of the same acquirer.
Also selects three case studies by fixed rules: largest flagged failure, largest missed failure, largest false alarm.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "losses_avoided"


def table(k, models):
    rows = []
    tot_imp = k["impair_next3"].sum()
    n_fail = k["fail"].sum()
    for m in models:
        flag = k[m] >= k[m + "_hi"] if m != "No tool" else pd.Series(False, index=k.index)
        f, nf = k[flag], k[~flag]
        rows.append({
            "rule": m,
            "deals flagged": int(flag.sum()),
            "% of deals flagged": flag.mean(),
            "failure rate if flagged": f["fail"].mean() if len(f) else np.nan,
            "failure rate if not flagged": nf["fail"].mean(),
            "failures caught (%)": f["fail"].sum() / n_fail,
            "impairments in flagged deals ($bn)": f["impair_next3"].sum() / 1e9,
            "impairments caught (%)": f["impair_next3"].sum() / tot_imp,
            "false alarms (good deals flagged)": int((f["fail"] == 0).sum()),
            "deal value of flagged failures ($bn)": f.loc[f["fail"] == 1, "deal_value"].sum() / 1e9,
            "deal value of false alarms ($bn)": f.loc[f["fail"] == 0, "deal_value"].sum() / 1e9,
        })
    return pd.DataFrame(rows).set_index("rule")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(ROOT / "outputs" / "final" / "predictions.csv")
    models = ["No tool", "Literature", "Logit (all)", "GBM"]
    tf = ROOT / "outputs" / "tuning" / "tuned_predictions.csv"
    if tf.exists():                                   # tuned models (tuning.py), same deals
        s = s.merge(pd.read_csv(tf), on=["cik", "fyear", "accn"], how="left", validate="1:1")
        models += ["Logit (all) tuned", "GBM tuned"]
    k = s[s["fail"].notna()].copy()
    k["impair_next3"] = k["impair_next3"].fillna(0)
    print(f"Final test, outcome known: {len(k):,} deals, {int(k['fail'].sum()):,} failures ({k['fail'].mean():.1%}), "
          f"goodwill impairments t+1..t+3: ${k['impair_next3'].sum() / 1e9:,.0f}bn")
    t = table(k, models)
    pd.set_option("display.width", 250)
    print(t.T.round(3).to_string())
    t.to_csv(OUT / "flag_rule_final.csv")

    # same rule on all out-of-sample years (development 2014-2016 + final 2017-2022)
    a = pd.concat([pd.read_csv(ROOT / "outputs" / p / "predictions.csv") for p in ("dev", "final")])
    a = a[a["fail"].notna()].copy()
    a["impair_next3"] = a["impair_next3"].fillna(0)
    ta = table(a, ["No tool", "Literature", "Logit (all)", "GBM"])
    print(f"\nAll out-of-sample years 2014-2022 ({len(a):,} deals):")
    print(ta.T.round(3).to_string())
    ta.to_csv(OUT / "flag_rule_2014_2022.csv")

    # ---- case studies, chosen by fixed rules (final test), for the tuned logit (the tool) and for GBM
    st = pd.read_csv(ROOT / "outputs" / "stock_test" / "stock_test_events.csv")[["cik", "fyear", "bhar36_raw"]]
    c = k.merge(st, on=["cik", "fyear"], how="left", validate="1:1")
    out = []
    for m in [x for x in ("Logit (all) tuned", "GBM") if x in c.columns]:
        c["risk_pct"] = c.groupby("fyear")[m].rank(pct=True)
        c["flag"] = c[m] >= c[m + "_hi"]
        cases = {
            "Hit: largest impairment among flagged deals": c[c["flag"] & (c["fail"] == 1)].nlargest(1, "impair_next3"),
            "Miss: largest impairment among deals not flagged": c[~c["flag"] & (c["fail"] == 1)].nlargest(1, "impair_next3"),
            "False alarm: largest flagged deal without failure": c[c["flag"] & (c["fail"] == 0)].nlargest(1, "deal_value"),
        }
        cols = ["name", "fyear", "deal_value", "acq_size", m, "risk_pct", "impair_next3", "impair_ratio", "bhar36_raw",
                "valuation", "gw_share", "stock_share", "sector"]
        cs = pd.concat({k2: v[cols].rename(columns={m: "probability"}) for k2, v in cases.items()}).droplevel(1)
        cs.insert(0, "model", m)
        print(f"\nCase studies (final test), {m}:")
        print(cs.T.to_string())
        out.append(cs)
    pd.concat(out).to_csv(OUT / "case_studies.csv")

if __name__ == "__main__":
    main()
