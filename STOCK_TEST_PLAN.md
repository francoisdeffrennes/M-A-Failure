# Additional stock-market test: declared before the data was obtained

Written 2026-11-25, before the Bloomberg prices of delisted acquirers were downloaded. This is a COMPLEMENTARY test,
added after the main (pre-registered) results were seen. It will be run once and reported as such.

## Why
In the main test, 30% of acquirers had no Yahoo prices (delisted, taken over, bankrupt). Their accounting failure rate
was 46% vs 23% for the others, so the stock-market test was biased toward survivors and had little power.

## Sample
All out-of-sample predictions already produced: acquisitions of fiscal years 2014-2022 (development walk-forward
2014-2016 and final walk-forward 2017-2022; the models and predictions are NOT changed).
Prices: Yahoo for listed companies; Bloomberg monthly total-return index for the others (historical tickers read
from their 10-K filings; a Bloomberg match is kept only if the company name matches the SEC name).

## Outcome
BHAR36 = buy-and-hold return of the acquirer minus buy-and-hold return of the US market (Fama-French) over the 36 months
starting the month after the acquisition-year 10-K is filed. If the stock stops trading (delisting), the proceeds are
reinvested in the market for the remaining months (standard treatment). At least one month of returns is required.
BHAR36 is winsorised at the 1st and 99th percentiles.

## Test (one specification)
OLS of BHAR36 on the GBM risk percentile (rank of the predicted failure probability within the fiscal year, 0 to 1),
with fiscal-year fixed effects; standard errors clustered by entry month.
Hypothesis: negative coefficient (higher predicted risk -> lower subsequent stock performance).
The coefficient is the expected BHAR difference between the riskiest and the safest deal. Two-sided p-value reported.

Reported alongside (descriptive, no additional test): mean BHAR36 by risk tercile, with and without the Bloomberg-added
companies, so the effect of correcting survivorship is visible.
