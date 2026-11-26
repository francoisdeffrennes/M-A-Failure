"""Step 2f - Read the Bloomberg file (monthly total-return index of acquirers that Yahoo does not cover),
keep only the series that belong to the right company, and save them as monthly returns.

Identity checks (a ticker can be re-used by another company after a delisting):
  1. name: the Bloomberg name must match the SEC name (same first significant word, or at least half of the
     significant words in common), after removing punctuation and legal suffixes (Inc, Corp, Holdings...)
  2. dates: checked later, event by event, in stock_test.py (the series must exist when the investor enters)
Input : data/bloomberg/bloomberg_prices.xlsx  (not published: Bloomberg licence)
Output: data/bloomberg/bbg_monthly_returns.csv (series end at the delisting) (columns = cik), outputs/bloomberg_name_check.csv
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUFFIX = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "HOLDINGS", "HOLDING", "HLDGS", "LTD",
          "LIMITED", "PLC", "LP", "L", "P", "LLC", "THE", "GROUP", "GRP", "CL", "CLASS", "A", "B", "SA", "NV", "SE",
          "AG", "DE", "NEW", "OLD", "INTL", "INTERNATIONAL", "AND", "OF", "TRUST", "PARTNERS", "ADR", "REDH", "ORD"}
# Manual review of the name test, done from the names and first price dates only, before any return was computed.
# Same company, renamed by its acquirer after the takeover (Bloomberg shows the post-deal name):
RENAMED = {"ALBANY MOLECULAR RESEARCH INC", "Consolidated Communications Holdings, Inc.",
           "Endurance International Group Holdings, Inc.", "ENERNOC INC", "FERRO CORP", "Houghton Mifflin Harcourt Co",
           "LOGMEIN, INC.", "NUMEREX CORP /PA/", "Raptor Pharmaceutical Corp", "STAMPS.COM INC", "Vista Outdoor Inc",
           "WHITEWAVE FOODS Co", "RICE MIDSTREAM PARTNERS LP", "NRC GROUP HOLDINGS CORP.", "SOC TELEMED, INC.",
           "BEACON ROOFING SUPPLY, INC.", "CEDAR FAIR L P", "ADESTO TECHNOLOGIES CORPORATION"}
# Different company that shares the first word:
NOT_SAME = {"Core-Mark Holding Company, LLC"}


def words(name):
    s = unicodedata.normalize("NFKD", str(name)).upper()
    s = re.sub(r"/[A-Z]{2}/?$", " ", s)                     # EDGAR state suffix, e.g. "/DE"
    s = s.replace("&", " AND ").replace("'", "")
    toks = re.sub(r"[^A-Z0-9]+", " ", s).split()
    return [t for t in toks if t not in SUFFIX]


def same_company(sec, bbg):
    if not isinstance(bbg, str) or bbg.startswith("#") or not bbg.strip():
        return False
    a, b = words(sec), words(bbg)
    if not a or not b:
        return False
    if a[0] == b[0] or "".join(a).startswith("".join(b)) or "".join(b).startswith("".join(a)):
        return True
    common = len(set(a) & set(b))
    return common / min(len(set(a)), len(set(b))) >= 0.5


def main():
    f = ROOT / "data" / "bloomberg" / "bloomberg_prices.xlsx"
    ck = pd.read_excel(f, "check")
    px = pd.read_excel(f, "prices", header=None)
    ck["name_ok"] = [same_company(s, b) for s, b in zip(ck["sec_name"], ck["bloomberg_name"])]
    ck["name_check"] = np.where(ck["name_ok"], "auto match", "no match")
    ck.loc[ck["sec_name"].isin(RENAMED), ["name_ok", "name_check"]] = [True, "renamed after takeover (manual)"]
    ck.loc[ck["sec_name"].isin(NOT_SAME), ["name_ok", "name_check"]] = [False, "different company (manual)"]
    series = {}
    for j in range(0, px.shape[1], 2):
        cik = int(px.iat[0, j + 1])
        d = pd.to_datetime(px.iloc[1:, j], errors="coerce")
        v = pd.to_numeric(px.iloc[1:, j + 1], errors="coerce")
        s = pd.Series(v.values, index=d.values).dropna()
        s = s[s.index.notna() & (s > 0)]
        if len(s) >= 2:
            s.index = pd.PeriodIndex(s.index, freq="M")
            s = s[~s.index.duplicated(keep="last")].sort_index()
            series[cik] = s
    ck["n_prices"] = ck["cik"].map(lambda c: len(series.get(int(c), [])))
    ck["first_price"] = ck["cik"].map(lambda c: str(series[c].index[0]) if c in series else None)
    ck["last_price"] = ck["cik"].map(lambda c: str(series[c].index[-1]) if c in series else None)
    keep = ck.loc[ck["name_ok"] & (ck["n_prices"] > 0), "cik"].astype(int)
    P = pd.DataFrame({c: series[c] for c in keep}).sort_index()
    R = P.pct_change(fill_method=None)
    R = R.where(R.abs() < 3)                                   # same data-error rule as the Yahoo returns
    # Bloomberg repeats the last price after a delisting (BDH fills non-trading days with the previous value), which
    # would show a delisted stock earning 0% a month until today. Cut each series after its last non-zero return:
    # the stock stopped trading there, and stock_test.py reinvests the proceeds in the market as declared.
    for c in R.columns:
        nz = R[c][R[c].fillna(0) != 0]
        if len(nz):
            R.loc[R.index > nz.index[-1], c] = np.nan
    R.index = R.index.astype(str)
    R.to_csv(ROOT / "data" / "bloomberg" / "bbg_monthly_returns.csv")
    ck.to_csv(ROOT / "outputs" / "bloomberg_name_check.csv", index=False)
    print(f"Bloomberg file: {len(ck)} companies | ticker given {ck['bloomberg_ticker'].notna().sum()} | "
          f"price series {(ck['n_prices'] > 0).sum()} | name matches {ck['name_ok'].sum()} | "
          f"kept (name match + prices) {len(keep)}")


if __name__ == "__main__":
    main()
