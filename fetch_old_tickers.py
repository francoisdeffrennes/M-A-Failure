"""Step 2d - Historical tickers of acquirers that Yahoo no longer covers (delisted, taken over, bankrupt).

The ticker printed in the acquisition-year 10-K (XBRL tag dei:TradingSymbol) is read directly from that
filing on EDGAR. These tickers are then used to pull prices of delisted companies from Bloomberg.
Output: outputs/missing_acquirers_for_bloomberg.csv (adds column old_ticker).
"""
import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "AI-for-Finance capstone student research contact@example.com"}
SYMBOL = re.compile(rb"(?<!No)TradingSymbol(?=[\"\s>])[^>]*>(?:\s*<[^>]+>)*\s*([A-Za-z][A-Za-z0-9.\-]{0,9})\s*<")   # tolerates nested tags


def get(url, first_bytes=None):
    """Download a URL; with first_bytes, only the beginning of the file (the cover page holds the ticker)."""
    h = dict(UA, **({"Range": f"bytes=0-{first_bytes}"} if first_bytes else {}))
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
        return r.read(first_bytes + 1 if first_bytes else -1)


def ticker_in_filing(cik, accn):
    folder = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn.replace('-', '')}"
    items = json.loads(get(folder + "/index.json"))["directory"]["item"]
    names = [i["name"] for i in items]
    # XBRL instance (.xml, not linkbases) first, then inline-XBRL .htm documents
    cands = [n for n in names if n.endswith(".xml") and not re.search(r"_(cal|def|lab|pre)\.xml$|FilingSummary", n)]
    size = {i["name"]: int(i["size"] or 0) for i in items}
    cands += sorted([n for n in names if n.endswith(".htm")], key=lambda n: -size[n])    # main 10-K document is the largest
    for n in cands[:4]:
        m = SYMBOL.search(get(f"{folder}/{n}", first_bytes=400_000))
        time.sleep(0.12)
        if m:
            t = m.group(1).decode().upper()
            if t not in ("TRUE", "FALSE", "NONE", "NA", "N"):
                return t
    return None


def latest_10k_accns(cik, n=2):
    """Accession numbers of the company's most recent 10-K filings (filed 2019+ carry the ticker on the cover)."""
    d = json.loads(get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"))["filings"]["recent"]
    acc = [a for a, f in zip(d["accessionNumber"], d["form"]) if f == "10-K"]
    return acc[:n]


def main():
    s = pd.concat([pd.read_csv(ROOT / "outputs" / p / "predictions.csv") for p in ("dev", "final")])
    px = pd.read_csv(ROOT / "data" / "prices" / "monthly_adjclose.csv", index_col=0, nrows=1)
    miss = s[~s["ticker"].isin(px.columns)].sort_values("filed")
    cache = ROOT / "outputs" / "old_tickers_cache.csv"            # saved after every company; the run can resume
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["cik"])
    rows = done.to_dict("records")
    seen = set(done["cik"].astype(int))
    for cik, g in miss.groupby("cik"):
        if int(cik) in seen:
            continue
        tick = None
        try:
            extra = latest_10k_accns(cik)
        except Exception:
            extra = []
        for accn in list(g["accn"].iloc[::-1]) + extra:          # acquisition-year 10-Ks, then latest 10-Ks
            try:
                tick = ticker_in_filing(cik, accn)
            except Exception:
                tick = None
            if tick:
                break
        rows.append(dict(cik=int(cik), name=g["name"].iloc[0], old_ticker=tick, first_deal_year=int(g["fyear"].min()),
                         last_deal_year=int(g["fyear"].max()), n_deals=len(g), last_10k_filed=g["filed"].max()))
        pd.DataFrame(rows).to_csv(cache, index=False)
        if len(rows) % 50 == 0:
            print(len(rows), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "missing_acquirers_for_bloomberg.csv", index=False)
    print(f"{len(out)} companies, ticker found for {out['old_ticker'].notna().sum()}")


if __name__ == "__main__":
    main()
