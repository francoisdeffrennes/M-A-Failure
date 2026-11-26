# Final audit (2026-11-26)

The whole project was re-run from the raw inputs in a clean directory (SEC archive, SEC metadata, Yahoo and
Kenneth French downloads, Bloomberg file). Every intermediate file and every output table and figure of the
pre-registered pipeline was reproduced exactly.

## Inputs checked
- SEC company-facts archive: 20,620 company files, none unparseable; extraction gives the same 2,958,885 facts.
- Kenneth French factors: identical to a fresh download (1,203 months).
- Yahoo prices: 14 of 15 randomly chosen tickers re-downloaded with identical monthly returns (one request rate-limited).
- SEC industry codes: 15 of 15 random companies identical to a fresh query.
- Labels spot-checked against company releases: Baker Hughes 2020 goodwill impairment $14,773m (exact);
  FIS 2022 impairment $17.6bn (pipeline: $17.7bn over 2020-2022).
- Citations checked: Kroll 2026 study, Lev and Gu (Fortune 2024), Gu and Lev (2011), Loughran and Vijh (1997),
  Moeller et al. (2005), FASB goodwill project (2018-2022).

## Problem found: label timing (look-ahead)
The pre-registered design trained the model for fiscal year Y on all deals of years <= Y-3. The outcome of a Y-3 deal
needs the company's fiscal-Y 10-K, which can be filed after some year-Y deals are reported. On average 4-10% of the
training labels (35-40% of the most recent cohort) were not yet public at the scored deal's filing date.

## Fix
`src/realtime.py` scores every deal with a model trained only on deals whose outcome was filed before the first day
of the month in which the deal's own 10-K was filed (tuning validation folds follow the same rule; years without a
usable validation fold keep the default settings). `src/corrected_results.py` recomputes every table on these
predictions; the manuscript reports them as the main results. `tests/test_pipeline.py` checks the rule.

## Effect
- AUC of all trained models: about 0.69-0.70 (pre-registered run: 0.69-0.73).
- Tuned logistic model vs literature benchmark: +0.004 [-0.028, +0.035] (pre-registered run: +0.034 [+0.005, +0.060]).
  The earlier advantage is not claimed. Restricting training to deals <= Y-4 (`src/robustness_lag.py`) gives the same
  conclusion (+0.005).
- Alert rule (tuned logistic): flags 37% of deals, catches 58% of failures and $117bn of $286bn of impairments
  (pre-registered run: 38%, 60%, $143bn).
- Stock tests: declared mean test unchanged (it was run once on the pre-registered predictions: -8.3 pts, p = 0.28);
  same specification on corrected predictions -7.5 pts (p = 0.32); exploratory median -23.0 pts (p = 0.004).

## Other fixes
- Numbers quoted in the manuscript that no script produced (Table 1 metrics for tuned models, market-label AUCs,
  share of acquirers trailing the market) are now written by `tuning.py`, `market_checks.py` and
  `corrected_results.py`.
- Presentation script shortened from 533 to 292 words to fit the strict two-minute limit.
- Manuscript wording: success metrics vs what was pre-registered, public-float definition of valuation, combined
  impairment tag, Bloomberg manual name review (18 companies), case-study deal values.
