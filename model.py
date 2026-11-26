"""Step 4 - Predicting failed acquisitions (walk-forward) and explaining the predictions.

Timing rule (no look-ahead): a prediction for an acquisition of fiscal year Y is made when the
acquirer's 10-K for Y is filed (early Y+1). At that date, the outcome of an older acquisition t is known
only if its three follow-up 10-Ks (t+1..t+3) have been filed, i.e. t <= Y-3. So the model that scores
year Y is trained on acquisitions of years <= Y-3 only.

Models
  Base rate     : the historical failure rate (no information; AUC = 0.5)
  Literature    : logistic regression on four drivers documented in research
                  (paid in shares, buyer valuation, deal size, goodwill share of price)
  Logit (all)   : logistic regression on all features
  GBM           : gradient-boosted trees on all features (fixed settings, never tuned on test years)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from features import CATEGORICAL, FEATURES

NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
LITERATURE = ["stock_share", "valuation", "acq_size", "gw_share"]
GBM_PARAMS = dict(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=30, l2_regularization=1.0,
                  random_state=0)
LAG = 3


class Prep:
    """Median imputation + missing flags + standardisation, fitted on training data only."""

    def fit(self, X, cols):
        self.cols = cols
        self.med = X[cols].median()
        Z = X[cols].fillna(self.med)
        self.mu, self.sd = Z.mean(), Z.std().replace(0, 1)
        self.flags = [c for c in cols if X[c].isna().any()]
        return self

    def transform(self, X):
        Z = (X[self.cols].fillna(self.med) - self.mu) / self.sd
        for c in self.flags:
            Z[c + "_missing"] = X[c].isna().astype(float)
        return Z.values


def sector_dummies(train, test):
    cats = sorted(train["sector"].dropna().unique())
    d = lambda X: np.column_stack([(X["sector"] == c).astype(float) for c in cats[1:]]) if len(cats) > 1 else np.empty((len(X), 0))
    return d(train), d(test)


def fit_predict(name, tr, te):
    """Predicted failure probabilities for `te`, using a model fitted on `tr`."""
    y = tr["fail"].values
    if name == "Base rate":
        return np.full(len(te), y.mean())
    if name in ("Literature", "Logit (all)"):
        cols = LITERATURE if name == "Literature" else NUMERIC
        prep = Prep().fit(tr, cols)
        Xtr, Xte = prep.transform(tr), prep.transform(te)
        if name == "Logit (all)":
            dtr, dte = sector_dummies(tr, te)
            Xtr, Xte = np.c_[Xtr, dtr], np.c_[Xte, dte]
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y)
        return m.predict_proba(Xte)[:, 1]
    if name == "GBM":
        cats = sorted(tr["sector"].dropna().unique())
        enc = lambda X: np.c_[X[NUMERIC].values, X["sector"].map({c: i for i, c in enumerate(cats)}).fillna(-1).values]
        m = HistGradientBoostingClassifier(categorical_features=[len(NUMERIC)], **GBM_PARAMS)
        m.fit(enc(tr), y)
        Xte = enc(te)
        Xte[:, -1] = np.where(Xte[:, -1] < 0, np.nan, Xte[:, -1])  # sectors unseen in training -> missing
        return m.predict_proba(Xte)[:, 1]
    raise ValueError(name)


MODELS = ["Base rate", "Literature", "Logit (all)", "GBM"]


def walk_forward(ev, years):
    """Out-of-sample probabilities for every event of `years` (one refit per year)."""
    out = []
    for Y in years:
        tr = ev[(ev["fyear"] <= Y - LAG) & ev["fail"].notna()]
        te = ev[ev["fyear"] == Y].copy()
        assert tr["fyear"].max() <= Y - LAG
        for name in MODELS:
            p = fit_predict(name, tr, pd.concat([tr, te]))
            p_tr, te[name] = p[:len(tr)], p[len(tr):]
            # risk-group cut points from the training predictions (known at prediction time)
            te[name + "_lo"], te[name + "_hi"] = np.quantile(p_tr, 1 / 3), np.quantile(p_tr, 2 / 3)
        te["n_train"] = len(tr)
        out.append(te)
    return pd.concat(out)


def scores(df, models=MODELS):
    """AUC, average precision, Brier score and failure rate in the riskiest fifth, per model."""
    d = df[df["fail"].notna()]
    rows = {}
    for m in models:
        p, y = d[m].values, d["fail"].values
        top = p >= np.quantile(p, 0.8)
        bottom = p <= np.quantile(p, 0.2)
        rows[m] = dict(AUC=roc_auc_score(y, p) if p.std() > 0 else 0.5, avg_precision=average_precision_score(y, p),
                       brier=brier_score_loss(y, p), fail_rate_top20=y[top].mean() if p.std() > 0 else np.nan,
                       fail_rate_bottom20=y[bottom].mean() if p.std() > 0 else np.nan)
    return pd.DataFrame(rows).T


def auc_diff_ci(df, a, b, n_boot=2000, seed=0):
    """AUC(a) - AUC(b) with a 95% bootstrap interval (resampling events)."""
    d = df[df["fail"].notna()]
    y, pa, pb = d["fail"].values, d[a].values, d[b].values
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    return roc_auc_score(y, pa) - roc_auc_score(y, pb), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def auc_ci(df, a, n_boot=2000, seed=0):
    """AUC of model `a` with a 95% bootstrap interval (resampling events). Above 0.5 = better than chance."""
    d = df[df["fail"].notna()]
    y, p = d["fail"].values, d[a].values
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), np.percentile(b, 2.5), np.percentile(b, 97.5)
