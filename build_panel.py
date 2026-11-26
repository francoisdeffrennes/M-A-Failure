"""Step 2 - Firm-year panel, acquisition events and failure labels.

From the 10-K facts (as first reported):
  * one row per company and fiscal year, values taken from THAT year's 10-K only
    (current-year figures, plus the prior-year comparatives printed in the same report);
  * an ACQUISITION EVENT is a fiscal year in which the company bought businesses for at least
    10% of its total assets at the start of the year, and the purchase created goodwill;
  * the event is a FAILURE (accounting definition) if, over the next three fiscal years, the company
    writes off at least 10% of its goodwill (goodwill impairment).

Output: data/panel_firmyears.parquet and data/events.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FACTS = ROOT / "data" / "sec" / "facts_10k.parquet"

EVENT_SIZE = 0.10          # acquisitions >= 10% of beginning-of-year assets (robustness: 0.05, 0.20)
FAIL_SHARE = 0.10          # impairments over t+1..t+3 >= 10% of goodwill at the end of year t
HORIZON = 3                # years after the acquisition year
MIN_ASSETS = 50e6          # drop very small companies (shells, micro caps)
FLOWS = {"gw_acquired", "gw_impair", "cash_acq", "consideration", "stock_acq", "revenue", "net_income",
         "op_income", "op_cash_flow"}
STOCKS = {"assets", "goodwill", "equity", "liabilities", "cash", "intangibles"}


def annual(df):
    """Keep instants and flows covering roughly one year (330 to 400 days)."""
    dur = (df["end"] - df["start"]).dt.days
    return df[df["start"].isna() | dur.between(330, 400)]


def firm_years(facts):
    gaap = annual(facts[~facts["var"].isin(["public_float", "shares_out"])])
    # fiscal period end of each 10-K = latest period end it reports
    rep = gaap.groupby("accn").agg(cik=("cik", "first"), name=("name", "first"), period_end=("end", "max"),
                                   filed=("filed", "min")).reset_index()
    # one 10-K per company and period end: the first one filed
    rep = rep.sort_values("filed").drop_duplicates(["cik", "period_end"], keep="first")
    gaap = gaap.merge(rep[["accn", "period_end"]], on="accn")

    gaap["lag"] = (gaap["period_end"] - gaap["end"]).dt.days         # 0 = current year, ~365 = prior year
    cur = gaap[gaap["lag"].abs() <= 7]
    pri = gaap[gaap["lag"].between(330, 400)]

    def pick(x):
        """One value per filing and variable: the preferred tag first; if the same tag appears several
        times (e.g. segment breakdowns), the largest absolute value, which is the consolidated total."""
        x = x.assign(absval=x["val"].abs()).sort_values(["accn", "var", "rank", "absval"], ascending=[True, True, True, False])
        return x.drop_duplicates(["accn", "var"]).pivot(index="accn", columns="var", values="val")

    wide_cur = pick(cur)
    wide_pri = pick(pri).add_suffix("_prev")
    panel = rep.set_index("accn").join(wide_cur).join(wide_pri).reset_index()

    dei = facts[facts["var"].isin(["public_float", "shares_out"])]
    dei = dei.groupby(["accn", "var"])["val"].max().unstack()
    panel = panel.join(dei, on="accn")

    m = panel["period_end"].dt.month
    panel["fyear"] = np.where(m >= 6, panel["period_end"].dt.year, panel["period_end"].dt.year - 1)
    panel = panel.sort_values(["cik", "fyear", "filed"]).drop_duplicates(["cik", "fyear"], keep="first")
    return panel.reset_index(drop=True)


def events(panel):
    p = panel.copy()
    # beginning-of-year assets: the prior-year figure printed in this 10-K, else last year's 10-K
    prev_row = p.groupby("cik")[["fyear", "assets", "goodwill"]].shift(1)
    consecutive = prev_row["fyear"] == p["fyear"] - 1
    p["assets_lag"] = p["assets_prev"].fillna(prev_row["assets"].where(consecutive))
    p["goodwill_lag"] = p["goodwill_prev"].fillna(prev_row["goodwill"].where(consecutive))
    gw_change = (p["goodwill"] - p["goodwill_lag"]).clip(lower=0)
    p["gw_added"] = p["gw_acquired"].fillna(gw_change)
    # deal size: cash paid for businesses + shares issued for them, or the goodwill created if larger.
    # ("consideration" tags are rarely used and sometimes misused, so they are not relied upon.)
    paid = p["cash_acq"].fillna(0) + p["stock_acq"].fillna(0).clip(lower=0)
    p["deal_value"] = np.maximum(paid, p["gw_added"].fillna(0))
    p["acq_size"] = p["deal_value"] / p["assets_lag"]

    # outcome: impairments over the next HORIZON fiscal years, from those years' own 10-Ks
    imp = p.set_index(["cik", "fyear"])["gw_impair"]
    exists = pd.Series(1, index=imp.index)
    fut_imp, fut_obs = [], []
    for k in range(1, HORIZON + 1):
        key = list(zip(p["cik"], p["fyear"] + k))
        fut_obs.append(exists.reindex(key).fillna(0).values)
        fut_imp.append(imp.reindex(key).fillna(0).clip(lower=0).values)   # no impairment reported = 0 if 10-K exists
    p["impair_next3"] = np.sum(fut_imp, axis=0)
    p["years_observed"] = np.sum(fut_obs, axis=0)
    p["impair_ratio"] = p["impair_next3"] / p["goodwill"]
    known_fail = p["impair_ratio"] >= FAIL_SHARE
    p["fail_acct"] = np.where(known_fail, 1.0, np.where(p["years_observed"] == HORIZON, 0.0, np.nan))

    ev = p[(p["acq_size"] >= EVENT_SIZE) & (p["gw_added"] > 0) & (p["assets_lag"] >= MIN_ASSETS)
           & (p["public_float"] > 0)                       # listed equity (drops subsidiaries filing their own 10-K)
           & p["fyear"].between(2010, 2022)].copy()
    # same group reporting twice (e.g. parent and listed sub with identical accounts): keep the larger float
    ev["dup_key"] = list(zip(ev["fyear"], (ev["goodwill"] / 1e6).round(), (ev["assets_lag"] / 1e6).round()))
    ev = ev.sort_values("public_float", ascending=False).drop_duplicates("dup_key").drop(columns="dup_key")
    return p, ev.sort_values(["fyear", "cik"]).reset_index(drop=True)


def main():
    facts = pd.read_parquet(FACTS)
    panel = firm_years(facts)
    panel, ev = events(panel)
    panel.to_parquet(ROOT / "data" / "panel_firmyears.parquet", index=False)
    ev.to_parquet(ROOT / "data" / "events.parquet", index=False)
    print(f"Firm-years: {len(panel):,} ({panel['cik'].nunique():,} companies) | acquisition events 2010-2022: {len(ev):,}")


if __name__ == "__main__":
    main()
