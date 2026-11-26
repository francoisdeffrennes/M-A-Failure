#!/usr/bin/env bash
# Reproduces every table and figure of the manuscript, in order. About 1-2 hours, mostly downloads.
# Steps marked (once) were run a single time for the study; --force re-runs them for reproduction and logs it.
set -euo pipefail
cd "$(dirname "$0")"

python src/extract_sec.py            # 1. SEC XBRL archive (downloaded if missing, ~1.4 GB) -> 10-K facts as first reported
python src/build_panel.py            # 2. firm-year panel and acquisition events
python src/fetch_meta.py             # 3a. SIC codes and tickers (SEC submissions API)
python src/fetch_prices.py           # 3b. Yahoo monthly prices and Fama-French market factor

python src/run_all.py --period dev   # 4a. development walk-forward 2014-2016
python src/run_all.py --period final --force   # 4b. locked final test 2017-2022 (once; --force = reproduction)

# 5. Delisted acquirers: historical tickers from their 10-Ks and a Bloomberg Excel template.
python src/fetch_old_tickers.py
python src/make_bloomberg_template.py
#    Manual step (licensed data, not in this repository): open outputs/bloomberg_template.xlsx on a Bloomberg
#    terminal, let the formulas fill, paste as values and save it as data/bloomberg/bloomberg_prices.xlsx.
if [ -f data/bloomberg/bloomberg_prices.xlsx ]; then
  python src/merge_bloomberg.py
else
  echo "No Bloomberg file: the stock test will use Yahoo prices only (survivorship-biased)."
fi

python src/stock_test.py --force     # 6a. declared stock-market test (once; --force = reproduction) -> Table 3
python src/stock_test_median.py      # 6b. exploratory median regression -> Table 3
python src/tuning.py                 # 7. hyperparameters chosen inside each training window -> Table 1
python src/losses_avoided.py         # 8. alert rule and case studies -> Table 2
python src/market_checks.py          # 9. descriptive market checks quoted in the manuscript
python src/robustness_lag.py         # 10. robustness: training on deals <= Y-4
python src/realtime.py               # 11. CORRECTED real-time walk-forward (main results after the audit)
python src/corrected_results.py      # 12. Tables 1-3 and case studies on the corrected predictions
python src/figures.py                # 13. Figures 1-5
python tests/test_pipeline.py        # 14. checks: no look-ahead, real-time labels, preprocessing, BHAR, tuning
