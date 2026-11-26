"""Step 1 - Extract the accounting data we need from the SEC bulk XBRL archive.

Source: https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip
(one JSON file per company with every number it has tagged in its filings since 2009).

We keep only annual reports (form 10-K) and, for every fact, the filing in which it appeared,
with its filing date. Later steps use only facts known at a given date (as first reported),
so restatements published afterwards cannot leak into the features.

Output: data/sec/facts_10k.parquet  (one row per company x tag x period x filing)
"""
import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "sec" / "companyfacts.zip"
OUT = ROOT / "data" / "sec" / "facts_10k.parquet"

# Our variable name -> XBRL tags, in order of preference (companies use different tags for the same item)
TAGS = {
    "assets":       ["Assets"],
    "goodwill":     ["Goodwill"],
    "gw_acquired":  ["GoodwillAcquiredDuringPeriod"],
    "gw_impair":    ["GoodwillImpairmentLoss", "GoodwillAndIntangibleAssetImpairment"],   # 2nd: includes other intangibles
    "cash_acq":     ["PaymentsToAcquireBusinessesNetOfCashAcquired", "PaymentsToAcquireBusinessesGross"],
    "consideration": ["BusinessCombinationConsiderationTransferred1", "BusinessAcquisitionCostOfAcquiredEntityPurchasePrice"],
    "stock_acq":    ["StockIssuedDuringPeriodValueAcquisitions",
                     "BusinessCombinationConsiderationTransferredEquityInterestsIssuedAndIssuable",
                     "BusinessAcquisitionCostOfAcquiredEntityEquityInterestsIssuedAndIssuable"],
    "equity":       ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "liabilities":  ["Liabilities"],
    "cash":         ["CashAndCashEquivalentsAtCarryingValue"],
    "intangibles":  ["IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"],
    "revenue":      ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                     "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "net_income":   ["NetIncomeLoss"],
    "op_income":    ["OperatingIncomeLoss"],
    "op_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
}
DEI = {"public_float": "EntityPublicFloat", "shares_out": "EntityCommonStockSharesOutstanding"}
WANTED = {t: v for v, ts in TAGS.items() for t in ts}


def facts_of(company):
    """Yield one dict per 10-K fact we care about."""
    cik = company.get("cik")
    name = company.get("entityName")
    facts = company.get("facts", {})
    for tag, var in WANTED.items():
        node = facts.get("us-gaap", {}).get(tag)
        if not node:
            continue
        for unit, rows in node["units"].items():
            if unit != "USD":
                continue
            for r in rows:
                if r.get("form") != "10-K":
                    continue
                yield dict(cik=cik, name=name, var=var, tag=tag, rank=TAGS[var].index(tag), start=r.get("start"),
                           end=r["end"], val=r["val"], filed=r["filed"], accn=r["accn"], fy=r.get("fy"), fp=r.get("fp"))
    for var, tag in DEI.items():
        node = facts.get("dei", {}).get(tag)
        if not node:
            continue
        for unit, rows in node["units"].items():
            for r in rows:
                if r.get("form") != "10-K":
                    continue
                yield dict(cik=cik, name=name, var=var, tag=tag, rank=0, start=None, end=r["end"], val=r["val"],
                           filed=r["filed"], accn=r["accn"], fy=r.get("fy"), fp=r.get("fp"))


URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"


def download():
    """The SEC asks for a User-Agent naming the requester; about 1.4 GB."""
    import shutil
    import urllib.request
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "AI-for-Finance capstone student research contact@example.com"})
    print("Downloading", URL, "(about 1.4 GB)...", flush=True)
    with urllib.request.urlopen(req, timeout=600) as r, open(ZIP, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def main():
    if not ZIP.exists():
        download()
    rows, n = [], 0
    with zipfile.ZipFile(ZIP) as z:
        for info in z.infolist():
            if not info.filename.endswith(".json"):
                continue
            try:
                company = json.loads(z.read(info))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            rows.extend(facts_of(company))
            n += 1
            if n % 2000 == 0:
                print(f"{n:,} companies read, {len(rows):,} facts kept", flush=True)
    df = pd.DataFrame(rows)
    for c in ["end", "start", "filed"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")   # some files store it as text
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df["fp"] = df["fp"].astype(str)
    df.to_parquet(OUT, index=False)
    print(f"Done: {n:,} companies, {len(df):,} facts, {df['cik'].nunique():,} companies with 10-K facts -> {OUT}")


if __name__ == "__main__":
    main()
