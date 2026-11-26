"""Step 2c - Monthly total-return prices of the acquirers (Yahoo Finance, adjusted close) and the
market and risk-free rate (Kenneth French data library).

Limitation: Yahoo only knows companies that are still listed under their current ticker, so acquirers
that were later taken over, went private or failed are missing (survivorship bias). The script records
which events have prices so this can be measured.

Outputs: data/prices/monthly_adjclose.csv (month-end x ticker), data/prices/ff_monthly.csv
"""
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "prices"
UA = {"User-Agent": "Mozilla/5.0"}
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1=1230768000&period2=1795000000&interval=1mo&events=div,split"
FF = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"


def yahoo_monthly(ticker):
    url = YAHOO.format(ticker.replace(".", "-"))
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        d = json.load(r)["chart"]["result"][0]
    ts = pd.to_datetime(d["timestamp"], unit="s")
    adj = d["indicators"]["adjclose"][0]["adjclose"]
    s = pd.Series(adj, index=ts, name=ticker).dropna()
    return s.groupby(s.index.to_period("M")).last()            # one value per calendar month


def french_monthly():
    with urllib.request.urlopen(urllib.request.Request(FF, headers=UA), timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    rows = [l.split(",") for l in lines if l[:6].strip().isdigit() and len(l.split(",")[0].strip()) == 6]
    df = pd.DataFrame(rows, columns=["month", "mkt_rf", "smb", "hml", "rf"])
    df["month"] = pd.PeriodIndex(df["month"].str.strip(), freq="M")
    return df.set_index("month").astype(float) / 100


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "data" / "sec" / "meta.csv")
    tickers = sorted({t.split("|")[0] for t in meta["tickers"].dropna() if t})
    cache = OUT / "monthly_adjclose.csv"
    have = pd.read_csv(cache, index_col=0) if cache.exists() else pd.DataFrame()
    series, failed = [have[c] for c in have.columns] if len(have) else [], []
    if len(have):
        series = [pd.Series(have[c].values, index=pd.PeriodIndex(have.index, freq="M"), name=c) for c in have.columns]
    for i, t in enumerate(x for x in tickers if x not in have.columns):
        try:
            series.append(yahoo_monthly(t))
        except Exception:
            failed.append(t)
        time.sleep(0.2)
        if (i + 1) % 200 == 0:
            print(f"{i + 1} tickers", flush=True)
    px = pd.concat(series, axis=1).sort_index()
    px.index = px.index.astype(str)
    px.to_csv(cache)
    ff = french_monthly()
    ff.index = ff.index.astype(str)
    ff.to_csv(OUT / "ff_monthly.csv")
    print(f"prices for {px.shape[1]:,} of {len(tickers):,} tickers ({len(failed)} failed); FF to {ff.index[-1]}")


if __name__ == "__main__":
    main()
