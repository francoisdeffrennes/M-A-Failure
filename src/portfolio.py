"""Step 5 - Economic value: investing in acquirers according to the predicted failure risk.

Each acquirer enters the portfolios the month after its acquisition-year 10-K is filed (the first date
the prediction exists) and is held for HOLD months. Every month we hold, equally weighted:
  All acquirers   : every acquirer currently in its holding window (the benchmark)
  Low risk        : those whose predicted probability is in the bottom third
  High risk       : those in the top third
  Avoid high risk : All acquirers minus the high-risk ones
  Low - High      : long low risk, short high risk
The thirds are cut at the 1/3 and 2/3 quantiles of the model's predictions on its own TRAINING
sample, so the classification of an event never depends on events filed after it.
Transaction cost: COST_BPS per unit traded when a stock enters or leaves a portfolio.
"""
import numpy as np
import pandas as pd

HOLD = 36
COST_BPS = 20


def monthly_returns(px):
    """Simple monthly returns from adjusted prices (columns = tickers, index = 'YYYY-MM')."""
    r = px.pct_change(fill_method=None)
    return r.where(r.abs() < 3)                                    # drop obvious data errors (>300% in a month)


def holdings(ev, prob_col, cut_lo, cut_hi):
    """Entry month, exit month and risk group of each event with a price series."""
    h = ev[["cik", "fyear", "ticker", "filed", prob_col, cut_lo, cut_hi]].dropna(subset=["ticker", prob_col]).copy()
    h["entry"] = (pd.to_datetime(h["filed"]).dt.to_period("M") + 1)
    h["exit"] = h["entry"] + HOLD - 1
    h["group"] = np.where(h[prob_col] <= h[cut_lo], "Low risk", np.where(h[prob_col] >= h[cut_hi], "High risk", "Middle"))
    return h


def portfolio_returns(h, rets, months):
    """Equal-weight monthly returns of each portfolio, net of entry/exit costs."""
    R = rets.copy()
    R.index = pd.PeriodIndex(R.index, freq="M")
    out = {k: [] for k in ["All acquirers", "Low risk", "High risk", "Avoid high risk"]}
    for m in months:
        live = h[(h["entry"] <= m) & (h["exit"] >= m)]
        r = live["ticker"].map(lambda t: R.at[m, t] if (t in R.columns and m in R.index) else np.nan)
        live = live.assign(r=r.values).dropna(subset=["r"])
        # cost: a stock pays COST_BPS in its first month (entry) and in its last month (exit)
        cost = COST_BPS / 1e4 * ((live["entry"] == m).astype(float) + (live["exit"] == m).astype(float))
        live = live.assign(net=live["r"] - cost)
        groups = {"All acquirers": live, "Low risk": live[live["group"] == "Low risk"],
                  "High risk": live[live["group"] == "High risk"], "Avoid high risk": live[live["group"] != "High risk"]}
        for k, g in groups.items():
            out[k].append(g["net"].mean() if len(g) else np.nan)
    df = pd.DataFrame(out, index=months)
    df["Low - High"] = df["Low risk"] - df["High risk"]
    return df


def bhar(ev, rets, mkt, months=HOLD, min_months=30):
    """Buy-and-hold return of the acquirer minus the market over `months` months from the entry month
    (the month after the acquisition-year 10-K is filed). NaN if fewer than `min_months` returns."""
    R = rets.copy()
    R.index = pd.PeriodIndex(R.index, freq="M")
    out = []
    for t, f in zip(ev["ticker"], ev["filed"]):
        if pd.isna(t) or t not in R.columns:
            out.append(np.nan)
            continue
        start = pd.Period(pd.Timestamp(f), freq="M") + 1
        window = pd.period_range(start, start + months - 1, freq="M")
        r = R[t].reindex(window)
        m = mkt.reindex(window)
        ok = r.notna() & m.notna()
        out.append((1 + r[ok]).prod() - (1 + m[ok]).prod() if ok.sum() >= min_months else np.nan)
    return np.array(out)


def stats(r, rf, mkt_rf):
    """Annualised performance and CAPM alpha (Newey-West t, 6 lags) of a monthly return series."""
    import statsmodels.api as sm
    d = pd.concat([r.rename("r"), rf.rename("rf"), mkt_rf.rename("mkt")], axis=1).dropna()
    is_ls = r.name == "Low - High"
    ex = d["r"] - (0 if is_ls else d["rf"])
    wealth = (1 + d["r"]).cumprod()
    fit = sm.OLS(ex, sm.add_constant(d["mkt"])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return dict(months=len(d), ann_return=(wealth.iloc[-1] ** (12 / len(d)) - 1), ann_vol=d["r"].std() * np.sqrt(12),
                sharpe=ex.mean() / ex.std() * np.sqrt(12), max_drawdown=(wealth / wealth.cummax() - 1).min(),
                alpha_ann=fit.params["const"] * 12, alpha_t=fit.tvalues["const"], beta=fit.params["mkt"])
