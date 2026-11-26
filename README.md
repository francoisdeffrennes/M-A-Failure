# Can AI Warn an Acquirer? Predicting goodwill-impairment failures in US acquisitions

Capstone project, AI for Finance (Prof. Josep Gisbert), Type A: free application.

**Question.** Can a machine-learning tool tell an acquirer, at the time of the deal, how likely it is that the
acquisition will end in a goodwill impairment (the accounting trace of synergies that did not materialise), and which
losses would such a warning have flagged?

**Answer (locked out-of-sample test, acquisitions of fiscal 2017-2022, 1,379 deals).**

| | Result |
| --- | --- |
| Ranking quality (AUC), regularised logistic model | 0.727 [0.697, 0.755] vs 0.694 for the literature benchmark (difference +0.034 [+0.005, +0.060]) |
| Failure rate, safest to riskiest fifth | 10% -> 15% -> 25% -> 33% -> 56% |
| Alert rule (flag the top third by past deals) | flags 38% of deals, catches 60% of failures and $143bn of $286bn of impairments |
| Stock market (2,067 deals, 2014-2022) | median riskiest vs safest: -27 pts over 3 years (p = 0.002, exploratory); declared mean test: -8 pts (p = 0.28) |

The manuscript gives the full analysis, the benchmarks and the limitations.

## Repository layout

```
src/                     all code, one script per step (each file starts with a description)
  extract_sec.py         1  SEC XBRL archive -> 10-K facts as first reported (no restatement look-ahead)
  build_panel.py         2  firm-year panel, acquisition events, failure labels
  fetch_meta.py          3a SIC codes and tickers (SEC submissions API)
  fetch_prices.py        3b Yahoo monthly prices, Fama-French market factor
  features.py               16 deal and buyer indicators + sector
  model.py                  models, walk-forward validation, metrics
  portfolio.py              calendar-time portfolios, BHAR
  run_all.py             4  walk-forward predictions, benchmarks, robustness, portfolios (dev and locked final)
  fetch_old_tickers.py   5  historical tickers of delisted acquirers (read from their own 10-Ks)
  make_bloomberg_template.py  Excel template that downloads their prices on a Bloomberg terminal
  merge_bloomberg.py        name and date checks, Bloomberg monthly returns
  stock_test.py          6  declared stock-market test (STOCK_TEST_PLAN.md)
  stock_test_median.py      exploratory median regression
  tuning.py              7  hyperparameters chosen by time-series CV inside each training window
  losses_avoided.py      8  alert rule for the acquirer, case studies
  figures.py             9  Figures 1-5
tests/test_pipeline.py      checks: no look-ahead, training-only preprocessing, BHAR with delisting, tuning
PREREGISTRATION.md          hypotheses, models and settings frozen before the final test (code hash 8a18c19d6666548c)
STOCK_TEST_PLAN.md          stock test declared before the Bloomberg data were downloaded
outputs/                    every result table (CSV), console logs and figures used in the manuscript
run_everything.sh           reproduces everything in order
```

## How to reproduce

In Google Colab: open `MA_failure_colab.ipynb` (File > Open notebook > GitHub) and run the cells.
On a computer:

```bash
git clone https://github.com/francoisdeffrennes/M-A-Failure.git
cd M-A-Failure
pip install -r requirements.txt
./run_everything.sh
```

`requirements.txt` pins the exact library versions used for the manuscript. With other versions everything runs,
but the gradient-boosting results can move slightly (for example, AUC 0.702 instead of 0.707 with an older
scikit-learn); the logistic models and the literature benchmark are unaffected.

Steps 1-4 need only free data and an internet connection (about 1.4 GB from the SEC). Everything runs on a laptop:
building the dataset takes about an hour, the models seconds, the tuning about two minutes.

### Data sources

| Data | Source | Licence |
| --- | --- | --- |
| Accounting facts (10-K, 2009-2026) | [SEC XBRL company facts](https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip) | public |
| Industry codes, tickers | SEC submissions API | public |
| Monthly prices of listed acquirers | Yahoo Finance chart API (downloaded 2026-11-25) | public, re-downloaded by the code |
| US market and risk-free returns | [Kenneth French data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | public |
| Monthly total-return index of delisted acquirers | Bloomberg (`TOT_RETURN_INDEX_GROSS_DVDS`) | licensed: **not included** |

The Bloomberg prices are not redistributed. `outputs/bloomberg_template.xlsx` re-downloads them on any Bloomberg
terminal (save the filled file as `data/bloomberg/bloomberg_prices.xlsx`). Without it, the stock test runs on Yahoo
prices only, which covers 63% of deals instead of 85% and is biased toward surviving acquirers.
Yahoo prices are re-downloaded at run time, so the stock-market figures can move slightly if Yahoo revises its history.

### One-shot tests

The final walk-forward test and the declared stock test were each run once; a lock file prevented re-runs and
`run_log.txt` records the code hash and every run. `run_everything.sh` passes `--force` so the results can be
reproduced; forced runs are logged as such.

## Timing rule (why there is no look-ahead)

A prediction for an acquisition of fiscal year Y is made when the acquirer files its 10-K for Y (early Y+1).
At that date only acquisitions of years <= Y-3 have a known three-year outcome, so each year's model is trained on
those only. Features use facts as first reported, preprocessing is fitted on training rows, alert thresholds come
from training predictions, and tuning uses validation years inside the training window. `tests/test_pipeline.py`
checks these rules.

## Use of generative AI

Claude (Anthropic) was used as a coding and writing assistant. See the manuscript's acknowledgement.
