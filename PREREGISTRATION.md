# Pre-registration (FROZEN before the final test)

Frozen 2026-11-25 18:23 UTC. Code hash `8a18c19d6666548c` (sha256 of src/*.py, first 16 characters).
Outcomes of acquisitions made in fiscal years 2017-2022 have NOT been looked at.

## Question
Can machine learning recognise, from information public when the acquirer's annual report for the acquisition year
is filed, which completed acquisitions will turn out to be failures (the buyer overpaid: expected synergies did not materialise)?

## Data (all free, downloaded by the code)
SEC XBRL company facts (10-K, as first reported), SEC submissions (SIC codes, tickers), Yahoo Finance monthly adjusted
prices, Kenneth French monthly market factor. US-listed acquirers, financial firms (SIC 6000-6999) excluded.

## Acquisition event
Fiscal year in which a company with listed equity (public float > 0) and >= $50m beginning assets acquired businesses
for >= 10% of beginning assets (cash paid + shares issued, or goodwill created if larger) and created goodwill.

## Failure definitions
- PRIMARY (accounting): goodwill impairments over fiscal years t+1..t+3 >= 10% of goodwill at end of t.
  Unknown if the company stops filing before t+3 without such an impairment (excluded; robustness: counted as success).
- Secondary (market, priced subsample, survivorship-biased): 36-month buy-and-hold return minus market <= -20%.
- Secondary (combined): failure under either definition.

## Validation
Walk-forward by fiscal year Y = 2017..2022; model for Y trained only on acquisitions of years <= Y-3
(the only outcomes known when year-Y reports are filed). Development: Y = 2014..2016.

## Models (settings fixed, not tuned)
Base rate; Literature logit (stock share, valuation, deal size, goodwill share); Logit on all 16 features + sector;
GBM (HistGradientBoosting, depth 3, lr 0.05, 200 iterations, min leaf 30, L2 1).

## Hypotheses
- H1 (primary): GBM out-of-sample AUC for accounting failure > 0.5 (95% bootstrap interval above 0.5).
  Development: 0.582 [0.535, 0.628].
- H2: GBM AUC > Literature AUC (bootstrap interval of the difference above 0). Development: +0.028 [-0.023, 0.077] (not significant).
- H3: the riskiest 20% of deals (GBM) account for more than 20% of impairment losses (scaled by assets). Development: 31%.
- H4 (economic): an investor avoiding high-risk acquirers earns a higher Sharpe / CAPM alpha than holding all acquirers,
  and Low-minus-High risk earns positive alpha (36-month holding, 20 bps per trade). Development: no evidence.
- H5: the market-based failure is hard to predict (AUC near 0.5); development: 0.52-0.53.
Robustness fixed in advance: unknown outcomes as successes; impairment threshold 25%.

## Specifications and changes made during development (disclosed)
Data cleaning after inspecting development events: dropped subsidiaries filing their own 10-K and duplicate groups;
stopped using the rarely-used, sometimes misused "consideration" tag for deal size; added a fallback impairment tag.
Fixed a portfolio coding error. Added the market-based and combined labels (planned with the user before any run),
the impairment-capture measure and AUC intervals after the first development run.
The author previously worked on unrelated projects using 2015-2026 US market data.

## Discipline
One final run, enforced by a lock file; the run log stores the code hash.
