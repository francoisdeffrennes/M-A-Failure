"""Step 2e - Excel template to pull, on a Bloomberg terminal, the monthly total-return index of acquirers that
Yahoo does not cover. Open the file on the terminal, wait for the formulas to fill, save as bloomberg_prices.xlsx.

Sheet 'check'  : one row per company: SEC name, Bloomberg ticker, Bloomberg name and an automatic OK/CHECK flag
                 (is the first word of the SEC name in the Bloomberg name?). A wrong or missing ticker can be typed
                 into column C; the price formulas read column C, so they update by themselves.
Sheet 'prices' : for each company, two columns (dates, monthly total-return index) from BDH
"""
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
# Tickers the 10-K search missed, entered by hand; they are checked by the Bloomberg name test like all the others
MANUAL = {1024725: "TEN", 1593034: "ENDP", 1604738: "AINC", 1622879: "SHWZ"}


def main():
    df = pd.read_csv(ROOT / "outputs" / "missing_acquirers_for_bloomberg.csv")
    df["old_ticker"] = df["old_ticker"].fillna(df["cik"].map(MANUAL))
    # companies with a ticker first, then those without one (optional: ticker to look up by hand)
    df = df.assign(_no=df["old_ticker"].isna()).sort_values(["_no", "name"]).drop(columns="_no").reset_index(drop=True)
    wb = Workbook()
    ck = wb.active
    ck.title = "check"
    ck.append(["cik", "sec_name", "bloomberg_ticker", "bloomberg_name", "match", "first_deal_year", "n_deals"])
    for c in ck[1]:
        c.font = Font(bold=True)
    yellow = PatternFill("solid", fgColor="FFF2CC")
    for i, r in df.iterrows():
        row = i + 2
        tick = f"{r['old_ticker']} US Equity" if pd.notna(r["old_ticker"]) else None
        ck.append([int(r["cik"]), r["name"], tick,
                   f'=IF(C{row}="","",BDP(C{row},"NAME"))',
                   f'=IF(C{row}="","NO TICKER",IF(ISNUMBER(SEARCH(LEFT(B{row},FIND(" ",B{row}&" ")-1),D{row})),"OK","CHECK"))',
                   int(r["first_deal_year"]), int(r["n_deals"])])
        if tick is None:
            ck.cell(row=row, column=3).fill = yellow
    ck.column_dimensions["B"].width = 42
    ck.column_dimensions["C"].width = 18
    ck.column_dimensions["D"].width = 42
    ck.freeze_panes = "A2"

    px = wb.create_sheet("prices")
    for i, r in df.iterrows():
        col = 1 + 2 * i
        start = f"{int(r['first_deal_year'])}0101"              # yyyymmdd: unambiguous in any Excel locale
        px.cell(row=1, column=col, value=f"=check!C{i + 2}")    # the ticker typed in the check sheet
        px.cell(row=1, column=col + 1, value=int(r["cik"]))
        # monthly total-return index (dividends reinvested), from the first deal year to today
        px.cell(row=2, column=col,
                value=f'=IF({get_column_letter(col)}1=0,"",BDH({get_column_letter(col)}1,'
                      f'"TOT_RETURN_INDEX_GROSS_DVDS","{start}","20261130","Per=M","Days=A"))')
    out = ROOT / "outputs" / "bloomberg_template.xlsx"
    wb.save(out)
    print(f"{len(df)} companies ({df['old_ticker'].notna().sum()} with a ticker) -> {out}")


if __name__ == "__main__":
    main()
