"""Step 9 - Figures for the manuscript and the presentation (outputs/figures/*.png).
Reads only saved outputs; run after run_all.py, stock_test.py, losses_avoided.py and tuning.py."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
C = {"Literature": "#8a8a8a", "Logit (all) tuned": "#257de0", "GBM": "#f47230"}
LABEL = {"Literature": "Literature benchmark (4 drivers)", "Logit (all) tuned": "AI tool: regularised logit (16 indicators)",
         "GBM": "Gradient boosting (16 indicators)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titlelocation": "left",
                     "legend.frameon": False})


def final_with_tuned():
    base = pd.read_csv(ROOT / "outputs" / "final" / "predictions.csv")
    tuned = pd.read_csv(ROOT / "outputs" / "tuning" / "tuned_predictions.csv")
    return base.merge(tuned, on=["cik", "fyear", "accn"], how="left", validate="1:1")


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_auc_by_year(k):
    models = ["Literature", "GBM", "Logit (all) tuned"]
    yrs = sorted(k["fyear"].unique())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.axhline(0.5, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(yrs[0], 0.507, "no skill (coin flip)", color=INK2, fontsize=8.5)
    for m in models:
        a = [roc_auc_score(g["fail"], g[m]) for _, g in k.groupby("fyear")]
        tot = roc_auc_score(k["fail"], k[m])
        ax.plot(yrs, a, color=C[m], lw=2, marker="o", ms=6, mec="white", mew=1.5,
                label=f"{LABEL[m]}  (all years {tot:.3f})", zorder=3 if "tuned" in m else 2)
    ax.set_ylim(0.45, 0.92)
    ax.set_xticks(yrs)
    ax.set_xlabel("Fiscal year of the acquisition (out-of-sample test years)")
    ax.set_ylabel("AUC")
    ax.grid(axis="y", color=GRID)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Out-of-sample ranking quality, every test year")
    save(fig, "fig1_auc_by_year.png")


def fig_quintiles(k, m="Logit (all) tuned"):
    q = pd.qcut(k[m].rank(method="first"), 5, labels=["1\nsafest", "2", "3", "4", "5\nriskiest"])
    r = k.groupby(q, observed=False)["fail"].mean()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    shades = ["#b6d5f8", "#84b9f3", "#519ced", "#2370cd", "#0f4a96"]
    bars = ax.bar(range(5), r.values, color=shades, width=0.7)
    for b, v in zip(bars, r.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.0%}", ha="center", color=INK, fontsize=10,
                bbox=dict(fc="white", ec="none", pad=1), zorder=4)
    base = k["fail"].mean()
    ax.axhline(base, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(-0.45, base + 0.012, f"average {base:.0%}", color=INK2, fontsize=8.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(r.index)
    ax.set_xlabel("Predicted risk quintile (AI tool)")
    ax.set_ylabel("Deals ending in goodwill impairment")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_ylim(0, max(r.values) + 0.1)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("Failure rate by predicted risk, test years 2017-2022")
    save(fig, "fig2_failure_by_quintile.png")
    return r


def fig_flag(k):
    models = ["Literature", "GBM", "Logit (all) tuned"]
    rows = []
    for m in models:
        f = k[m] >= k[m + "_hi"]
        imp = k["impair_next3"].fillna(0)
        rows.append([f.mean(), k.loc[f, "fail"].sum() / k["fail"].sum(), imp[f].sum() / imp.sum()])
    v = np.array(rows)
    groups = ["Deals flagged", "Failures caught", "Impairment $ caught"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.26
    for i, m in enumerate(models):
        x = np.arange(3) + (i - 1) * (w + 0.02)
        bars = ax.bar(x, v[i], width=w, color=C[m], label=LABEL[m])
        for b, val in zip(bars, v[i]):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.012, f"{val:.0%}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(groups)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Alert rule: flag a deal when its risk is in the top third of past deals")
    save(fig, "fig4_flag_rule.png")
    return pd.DataFrame(v, index=models, columns=groups)


def fig_bhar():
    t = pd.read_csv(ROOT / "outputs" / "stock_test" / "bhar_by_tercile.csv")
    t = t[t["sample"] == "All (Yahoo + Bloomberg)"].set_index("tercile").loc[["Low risk", "Middle", "High risk"]]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(3)
    w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, t["median"], width=w, color="#257de0", label="Median deal")
    b2 = ax.bar(x + w / 2 + 0.01, t["mean"], width=w, color="#b6d5f8", label="Mean")
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v - 0.025, f"{v:+.0%}", ha="center", va="top", fontsize=9, color=INK)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["Low risk", "Middle", "High risk"])
    ax.set_xlabel("Predicted risk tercile (GBM, declared test)")
    ax.set_ylabel("3-year return vs US market")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_ylim(-0.48, 0.05)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("Acquirers' stock performance after the deal, 2014-2022 (n = 2,067)")
    save(fig, "fig5_bhar_by_tercile.png")


def fig_drivers():
    s = pd.concat([pd.read_csv(ROOT / "outputs" / p / "predictions.csv") for p in ("dev", "final")])
    k = s[s["fail"].notna()]
    spec = [("valuation", "Buyer valuation (market / book)", "high", "low"),
            ("ocf_assets", "Operating cash flow / assets", "high", "low"),
            ("roa", "Profitability (ROA)", "high", "low"),
            ("log_assets", "Buyer size", "large", "small"),
            ("acq_size", "Deal size / buyer assets", "small", "large"),
            ("stock_share", "Paid in shares", "cash", "shares")]
    rows = []
    for f, lab, good, bad in spec:
        q = pd.qcut(k[f].rank(method="first"), 5, labels=False)
        r = k.groupby(q)["fail"].mean()
        lo, hi = (r.iloc[-1], r.iloc[0]) if good in ("high", "large") else (r.iloc[0], r.iloc[-1])
        rows.append((f"{lab}\n({good} vs {bad})", lo, hi))
    p = k.groupby(k["prior_impair"] > 0)["fail"].mean()
    rows.append(("Impairment in the 3 prior years\n(no vs yes)", p[False], p[True]))
    sec = k.groupby("sector")["fail"].agg(["mean", "count"])
    sec = sec[sec["count"] >= 40]["mean"]
    rows.append((f"Sector\n({sec.idxmin()} vs {sec.idxmax()})", sec.min(), sec.max()))
    rows = sorted(rows, key=lambda r: r[2] - r[1])
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for i, (lab, lo, hi) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color="#cfcfcf", lw=3, solid_capstyle="round", zorder=1)
        ax.scatter([lo], [i], s=70, color="#84b9f3", edgecolor="white", lw=1.5, zorder=2)
        ax.scatter([hi], [i], s=70, color="#0f4a96", edgecolor="white", lw=1.5, zorder=2)
        ax.text(hi + 0.015, i, f"{hi:.0%}", va="center", fontsize=8.5, color=INK)
        ax.text(lo - 0.015, i, f"{lo:.0%}", va="center", ha="right", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_xlim(0, 0.7)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set_xlabel("Share of deals ending in goodwill impairment (safest vs riskiest fifth)")
    ax.scatter([], [], s=70, color="#84b9f3", label="favourable value")
    ax.scatter([], [], s=70, color="#0f4a96", label="unfavourable value")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("What drives the risk score (2,110 test deals, 2014-2022)")
    save(fig, "fig3_drivers.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = final_with_tuned()
    k = d[d["fail"].notna()]
    fig_auc_by_year(k)
    print("Failure rate by quintile (tuned logit):", fig_quintiles(k).round(3).to_dict())
    print(fig_flag(k).round(3).to_string())
    fig_bhar()
    fig_drivers()
    print("Figures saved to", OUT)


if __name__ == "__main__":
    main()
