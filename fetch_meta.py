"""Step 2b - Industry code (SIC) and current tickers of the acquirers, from the SEC submissions API.

Output: data/sec/meta.csv (cik, sic, sic_description, tickers, exchanges). Results are cached, so the
script only downloads companies it has not seen before. The SEC asks for at most 10 requests per second
and a User-Agent that identifies the requester.
"""
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sec" / "meta.csv"
UA = {"User-Agent": "AI-for-Finance capstone student research contact@example.com"}


def fetch(cik):
    url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        d = json.load(r)
    return dict(cik=int(cik), sic=d.get("sic"), sic_description=d.get("sicDescription"),
                tickers="|".join(d.get("tickers") or []), exchanges="|".join(x or "" for x in (d.get("exchanges") or [])))


def main():
    ciks = pd.read_parquet(ROOT / "data" / "events.parquet")["cik"].dropna().astype(int).unique()
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["cik"])
    todo = [c for c in ciks if c not in set(done["cik"])]
    rows = []
    for i, c in enumerate(todo):
        for attempt in range(3):
            try:
                rows.append(fetch(c))
                break
            except Exception:
                time.sleep(2)
        time.sleep(0.12)                                    # stay under 10 requests per second
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{len(todo)}", flush=True)
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"meta for {len(out):,} companies ({len(todo) - len(rows)} failed)")


if __name__ == "__main__":
    main()
