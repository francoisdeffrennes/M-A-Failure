"""Step 6b - EXPLORATORY (added after seeing the declared test): median regression of BHAR36 on the GBM risk
percentile with fiscal-year fixed effects. Motivated by the descriptive tercile medians; reported as exploratory,
next to the declared mean test, never instead of it. SE: bootstrap resampling entry months (clusters), 1,000 draws."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stock_test"


def main():
    e = pd.read_csv(OUT / "stock_test_events.csv").dropna(subset=["bhar36_raw"])
    lo, hi = e["bhar36_raw"].quantile([0.01, 0.99])
    e["bhar36"] = e["bhar36_raw"].clip(lo, hi)
    e["entry_m"] = (pd.to_datetime(e["filed"]).dt.to_period("M") + 1).astype(str)
    f = "bhar36 ~ risk_pct + C(fyear)"
    b = smf.quantreg(f, e).fit(q=0.5).params["risk_pct"]
    rng = np.random.default_rng(0)
    groups = {k: g for k, g in e.groupby("entry_m")}
    keys = list(groups)
    draws = []
    for _ in range(1000):
        s = pd.concat([groups[k] for k in rng.choice(keys, len(keys))], ignore_index=True)
        try:
            draws.append(smf.quantreg(f, s).fit(q=0.5, max_iter=2000).params["risk_pct"])
        except Exception:
            pass
    draws = np.array(draws)
    se = draws.std()
    ci = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws >= 0).mean(), (draws <= 0).mean())
    print(f"[Exploratory] median BHAR36 on GBM risk percentile, year FE: coef {b:+.3f}, bootstrap SE {se:.3f}, "
          f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], bootstrap p = {p:.3f}, N = {len(e):,}, draws {len(draws)}")
    pd.DataFrame([dict(coef=b, se=se, ci_lo=ci[0], ci_hi=ci[1], p_boot=p, n=len(e))]).to_csv(
        OUT / "exploratory_median_test.csv", index=False)


if __name__ == "__main__":
    main()
