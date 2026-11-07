"""Figures for the readout."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
INK, ACCENT, MUTED = "#1f2a36", "#b5452c", "#8a96a3"

rank = pd.read_csv(RES / "ranking_test.csv").set_index("model")
pred = json.loads((RES / "predictions.json").read_text())

# 1. AP by split and by model
fig, ax = plt.subplots(figsize=(6.4, 3.0))
labels = ["M1 random\n5-fold CV", "M1 customer-\ngrouped CV", "M0 logistic", "M1 XGBoost\nraw",
          "M2 + history", "M3 isolation\nforest", "M4 = M2 +\nanomaly score"]
vals = [pred["P1"]["random_cv_ap"], pred["P1"]["grouped_cv_ap"]] + [rank.loc[m, "ap"] for m in
                                                                    ["M0", "M1", "M2", "M3", "M4"]]
err_lo = [0, 0] + [rank.loc[m, "ap"] - rank.loc[m, "ap_lo"] for m in ["M0", "M1", "M2", "M3", "M4"]]
err_hi = [0, 0] + [rank.loc[m, "ap_hi"] - rank.loc[m, "ap"] for m in ["M0", "M1", "M2", "M3", "M4"]]
colors = [MUTED, MUTED, INK, INK, INK, INK, INK]
ax.bar(range(7), vals, color=colors, yerr=[err_lo, err_hi], capsize=3, width=0.65)
for i, v in enumerate(vals):
    ax.text(i, v + err_hi[i] + 0.02, f"{v:.2f}", ha="center")
ax.axvline(1.5, color=MUTED, lw=0.8, ls=":")
ax.set_xticks(range(7), labels, fontsize=7.5)
ax.set_ylabel("Average precision")
ax.set_ylim(0, 1)
ax.set_title("Grey: scored inside days 1-151. Dark: trained on days 1-121, scored on days 152-182",
             fontsize=8, loc="left")
fig.tight_layout()
fig.savefig(FIG / "fig1_ap_by_split_and_model.png", dpi=200, metadata={"Software": None})

# 2. cost under each decision rule
pol = pd.read_csv(RES / "policies_test.csv")
fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9), sharey=True)
names = {"none": "none", "T1_f1_cutoff": "T1\nF1", "T2_cost_cutoff": "T2\ncost",
         "T3_bayes_min_risk": "T3\np x amt"}
for ax, c in zip(axes, [5.0, 10.0, 25.0]):
    sub = pol[(pol.action_cost == c) & pol.rule.isin(names)].set_index("rule").loc[list(names)]
    ax.bar(range(4), sub.total_cost / 1000,
           yerr=[(sub.total_cost - sub.cost_lo) / 1000, (sub.cost_hi - sub.total_cost) / 1000],
           color=[MUTED, INK, INK, ACCENT], capsize=3, width=0.65)
    ax.set_xticks(range(4), names.values(), fontsize=7)
    ax.set_title(f"Review cost ${c:.0f} per alert", fontsize=8.5)
axes[0].set_ylabel("Test-month cost ($ thousands)")
fig.tight_layout()
fig.savefig(FIG / "fig2_cost_by_rule.png", dpi=200, metadata={"Software": None})

# 3. week by week
wk = pd.read_csv(RES / "weekly_drift.csv")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 3.8), sharex=True)
for ax, col, label in [(ax1, "ap_M2", "M2 average\nprecision"), (ax2, "o_e", "Observed /\nexpected frauds")]:
    ax.plot(wk.first_day, wk[col], "o-", color=INK, ms=4)
    ax.axvline(151.5, color=MUTED, lw=0.8, ls=":")
    ax.set_ylabel(label, fontsize=8)
ax1.text(152.5, ax1.get_ylim()[1], "test month", fontsize=7.5, va="top", color=MUTED)
ax2.axhline(1, color=ACCENT, lw=0.8, ls="--")
ax2.set_xlabel("First day of week (weeks restart on day 152; the last week of each month is short)",
               fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig3_weekly_drift.png", dpi=200, metadata={"Software": None})
print("figures written")
