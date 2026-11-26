"""Step 3 - Features of each acquisition event.

Everything comes from the acquirer's 10-K for the acquisition year t (and earlier 10-Ks), so every
feature is public on the 10-K filing date. That date is when a prediction could first be made.

Deal
  acq_size      deal value / total assets at the start of the year
  gw_share      goodwill created / deal value: the part of the price paid above identifiable net
                assets, i.e. what was paid for expected synergies and growth
  stock_share   share of the price paid with the acquirer's own shares
Acquirer
  log_assets, leverage, cash_assets, roa, ocf_assets, op_margin, rev_growth,
  gw_assets     goodwill / total assets after the deal
  gw_equity     goodwill / book equity (a thin equity cushion makes impairments more likely and visible)
  intang_assets other intangibles / total assets
  valuation     log(public float / book equity): richly valued buyers tend to overpay (Gu & Lev 2011)
  prior_impair  goodwill impairment in years t-2..t
  serial        number of other large acquisition years in t-3..t-1
  sector        SIC division (categorical)
"""
import numpy as np
import pandas as pd

FEATURES = ["acq_size", "gw_share", "stock_share", "log_assets", "leverage", "cash_assets", "roa", "ocf_assets",
            "op_margin", "rev_growth", "gw_assets", "gw_equity", "intang_assets", "valuation", "prior_impair", "serial",
            "sector"]
CATEGORICAL = ["sector"]


def sic_division(sic):
    """Broad SIC divisions."""
    if pd.isna(sic):
        return "Unknown"
    s = int(sic)
    for lo, hi, name in [(100, 999, "Agri/Mining"), (1000, 1499, "Agri/Mining"), (1500, 1799, "Construction"),
                         (2000, 2799, "Manuf. non-durable"), (2800, 2899, "Chemicals/Pharma"), (2900, 3399, "Manuf. non-durable"),
                         (3400, 3569, "Manuf. durable"), (3570, 3579, "Tech hardware"), (3580, 3659, "Manuf. durable"),
                         (3660, 3699, "Tech hardware"), (3700, 3839, "Manuf. durable"), (3840, 3859, "Medical devices"),
                         (3860, 3999, "Manuf. durable"), (4000, 4799, "Transport"), (4800, 4899, "Telecom/Media"),
                         (4900, 4999, "Utilities"), (5000, 5999, "Trade/Retail"), (6000, 6999, "Financial"),
                         (7370, 7379, "Software/IT services"), (7000, 7369, "Services"), (7380, 7999, "Services"),
                         (8000, 8099, "Health services"), (8100, 8999, "Services")]:
        if lo <= s <= hi:
            return name
    return "Other"


def safe_div(a, b):
    b = b.where(b != 0)
    return a / b


def build(events, panel, meta):
    e = events.copy()
    liab = e["liabilities"].fillna(e["assets"] - e["equity"])
    paid = e["cash_acq"].fillna(0) + e["stock_acq"].fillna(0).clip(lower=0)
    e["acq_size"] = e["acq_size"].clip(upper=3)
    e["gw_share"] = safe_div(e["gw_added"], e["deal_value"]).clip(0, 1.5)
    e["stock_share"] = safe_div(e["stock_acq"].fillna(0).clip(lower=0), paid).clip(0, 1)
    e["log_assets"] = np.log(e["assets_lag"])
    e["leverage"] = safe_div(liab, e["assets"]).clip(0, 2)
    e["cash_assets"] = safe_div(e["cash"], e["assets"]).clip(0, 1)
    e["roa"] = safe_div(e["net_income"], e["assets_lag"]).clip(-1, 1)
    e["ocf_assets"] = safe_div(e["op_cash_flow"], e["assets_lag"]).clip(-1, 1)
    e["op_margin"] = safe_div(e["op_income"], e["revenue"]).clip(-2, 1)
    e["rev_growth"] = (safe_div(e["revenue"], e["revenue_prev"]) - 1).clip(-1, 5)
    e["gw_assets"] = safe_div(e["goodwill"], e["assets"]).clip(0, 1)
    e["gw_equity"] = safe_div(e["goodwill"], e["equity"].where(e["equity"] > 0)).clip(0, 10)
    e["intang_assets"] = safe_div(e["intangibles"], e["assets"]).clip(0, 1)
    e["valuation"] = np.log(safe_div(e["public_float"], e["equity"].where(e["equity"] > 0))).clip(-5, 5)

    # history from earlier 10-Ks of the same company (all filed before the event's 10-K)
    hist = panel[["cik", "fyear", "gw_impair", "acq_size"]].copy()
    past_imp, past_acq = [], []
    for k in (1, 2):
        h = hist.assign(fyear=hist["fyear"] + k).set_index(["cik", "fyear"])["gw_impair"]
        past_imp.append(h.reindex(list(zip(e["cik"], e["fyear"]))).fillna(0).values)
    for k in (1, 2, 3):
        h = hist.assign(fyear=hist["fyear"] + k).set_index(["cik", "fyear"])["acq_size"]
        past_acq.append((h.reindex(list(zip(e["cik"], e["fyear"]))) >= 0.05).values)
    e["prior_impair"] = ((np.sum(past_imp, axis=0) + e["gw_impair"].fillna(0).clip(lower=0)) > 0).astype(float)
    e["serial"] = np.sum(past_acq, axis=0).astype(float)

    e = e.merge(meta[["cik", "sic", "tickers"]], on="cik", how="left")
    e["sector"] = e["sic"].map(sic_division)
    return e
