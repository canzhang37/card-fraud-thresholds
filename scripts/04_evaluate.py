"""Test-month scores, dollar costs, verdicts P1-P5."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraudcost import metrics  # noqa: E402

RAW, RES = ROOT / "results" / "raw", ROOT / "results"
MODELS = ["M0", "M1", "M2", "M3", "M4", "S1"]
COSTS = [5.0, 10.0, 25.0]
B = 1000

df = pd.read_parquet(ROOT / "data" / "interim" / "full.parquet",
                     columns=["TransactionID", "isFraud", "TransactionAmt", "day", "split", "uid"])
df = df[df.split != "train"]
for m in MODELS:
    s = pd.read_parquet(RAW / f"scores_{m}.parquet").rename(columns={"score": m})
    df = df.merge(s, on="TransactionID", how="left")
assert df[MODELS].notna().all().all()
val, test = df[df.split == "valid"], df[df.split == "test"].reset_index(drop=True)
y, a = test["isFraud"].to_numpy(float), test["TransactionAmt"].to_numpy()
W = metrics.cluster_weights(test["uid"].to_numpy(), B, seed=7)


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def boot_ap(score):
    order = np.argsort(-score, kind="mergesort")
    s, yy = score[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    out = np.empty(B)
    for b in range(B):
        w = W[b, order]
        tp, seen = np.cumsum(w * yy)[last], np.cumsum(w)[last]
        # zero-weight leading rows add nothing
        prec = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
        out[b] = np.sum(np.diff(np.r_[0.0, tp]) * prec) / tp[-1]
    return out


# Ranking
rank_rows, ap_boot = [], {}
for m in MODELS:
    sc = test[m].to_numpy()
    ap_boot[m] = boot_ap(sc)
    row = {"model": m, "ap": metrics.average_precision(y, sc), "ap_lo": ci(ap_boot[m])[0],
           "ap_hi": ci(ap_boot[m])[1], "roc_auc": roc_auc_score(y, sc)}
    for r in (0.01, 0.02, 0.05):
        rec, rec_usd = metrics.recall_at(y, sc, a, r)
        row[f"recall_{int(r * 100)}pct"], row[f"usd_recall_{int(r * 100)}pct"] = rec, rec_usd
    rank_rows.append(row)
ranking = pd.DataFrame(rank_rows)
ranking.to_csv(RES / "ranking_test.csv", index=False, float_format="%.4f")

diffs = {}
for name, (m1, m2) in {"M2-M1": ("M2", "M1"), "M4-M2": ("M4", "M2"), "S1-M2": ("S1", "M2")}.items():
    d = ap_boot[m1] - ap_boot[m2]
    point = metrics.average_precision(y, test[m1]) - metrics.average_precision(y, test[m2])
    diffs[name] = {"diff": point, "lo": ci(d)[0], "hi": ci(d)[1]}

# isotonic calibration on the validation month
cal = {}
for m in ("M2", "S1"):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(val[m], val["isFraud"])
    cal[m] = iso.predict(test[m])
p = cal["M2"]
oe_boot = (W @ y) / (W @ p)
calib = {"brier_raw": float(np.mean((test["M2"] - y) ** 2)), "brier_isotonic": float(np.mean((p - y) ** 2)),
         "observed": float(y.sum()), "expected": float(p.sum()), "o_e": float(y.sum() / p.sum()),
         "o_e_lo": ci(oe_boot)[0], "o_e_hi": ci(oe_boot)[1]}
bins = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
reliability = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin").agg(
    transactions=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean")).reset_index()
reliability.to_csv(RES / "calibration_test.csv", index=False, float_format="%.4f")

# Decision rules on M2 scores, in dollars
t1 = metrics.f1_threshold(val["isFraud"].to_numpy(float), val["M2"].to_numpy())
policy_rows, policy_cmp = [], []
for c in COSTS:
    t2 = metrics.cost_threshold(val["isFraud"].to_numpy(float), val["M2"].to_numpy(),
                                val["TransactionAmt"].to_numpy(), c)
    flags = {"none": np.zeros(len(y), bool), "T1_f1_cutoff": test["M2"].to_numpy() >= t1,
             "T2_cost_cutoff": test["M2"].to_numpy() >= t2, "T3_bayes_min_risk": p * a >= c,
             "S1_T3_label_delay": cal["S1"] * a >= c}
    per_row = {k: c * f + (~f) * y * a for k, f in flags.items()}
    boots = {k: W @ v for k, v in per_row.items()}
    for k, f in flags.items():
        policy_rows.append({"action_cost": c, "rule": k, "total_cost": per_row[k].sum(),
                            "cost_per_1000_tx": 1000 * per_row[k].sum() / len(y),
                            "alert_rate": f.mean(), "frauds_caught": (f * y).sum() / y.sum(),
                            "fraud_usd_caught": (f * y * a).sum() / (y * a).sum(),
                            "cost_lo": ci(boots[k])[0], "cost_hi": ci(boots[k])[1]})
    for x, z in [("T3_bayes_min_risk", "T1_f1_cutoff"), ("T3_bayes_min_risk", "T2_cost_cutoff"),
                 ("S1_T3_label_delay", "T3_bayes_min_risk")]:
        d = boots[x] - boots[z]
        policy_cmp.append({"action_cost": c, "comparison": f"{x} - {z}",
                           "diff": per_row[x].sum() - per_row[z].sum(), "lo": ci(d)[0], "hi": ci(d)[1],
                           "pct_change": (per_row[x].sum() - per_row[z].sum()) / per_row[z].sum()})
pd.DataFrame(policy_rows).to_csv(RES / "policies_test.csv", index=False, float_format="%.4f")
pd.DataFrame(policy_cmp).to_csv(RES / "policy_comparisons_test.csv", index=False, float_format="%.4f")

# weekly drift, validation and test months
iso_m2 = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(val["M2"], val["isFraud"])
df["p_cal"] = iso_m2.predict(df["M2"])
# weeks restart at day 152
df["week"] = np.where(df["split"] == "valid", (df["day"] - 122) // 7, 10 + (df["day"] - 152) // 7)
weekly = df.groupby("week").apply(lambda g: pd.Series({
    "first_day": g.day.min(), "last_day": g.day.max(), "split": g.split.iloc[0],
    "transactions": len(g), "fraud_rate": g.isFraud.mean(),
    "ap_M2": metrics.average_precision(g.isFraud.to_numpy(float), g.M2.to_numpy()),
    "o_e": g.isFraud.sum() / g.p_cal.sum()}), include_groups=False).reset_index()
weekly.to_csv(RES / "weekly_drift.csv", index=False, float_format="%.4f")

# Predictions
cv = {k: json.loads((RAW / f"cv_{k}.json").read_text())["ap"] for k in ("random", "grouped")}
ap = dict(zip(ranking.model, ranking.ap))
p4 = next(r for r in policy_cmp if r["action_cost"] == 10.0 and r["comparison"].endswith("T1_f1_cutoff"))
verdicts = {
    "P1": {"random_cv_ap": cv["random"], "grouped_cv_ap": cv["grouped"], "time_test_ap": ap["M1"],
           "met": bool(cv["random"] - ap["M1"] >= 0.05
                       and min(cv["random"], ap["M1"]) <= cv["grouped"] <= max(cv["random"], ap["M1"]))},
    "P2": {**diffs["M2-M1"], "met": diffs["M2-M1"]["lo"] > 0},
    "P3": {"M3_ap": ap["M3"], "M1_ap": ap["M1"], **{f"M4-M2_{k}": v for k, v in diffs["M4-M2"].items()},
           "met": bool(ap["M3"] < 0.5 * ap["M1"] and abs(diffs["M4-M2"]["diff"]) < 0.01
                       and diffs["M4-M2"]["lo"] <= 0 <= diffs["M4-M2"]["hi"])},
    "P4": {**p4, "met": p4["hi"] < 0},
    "P5": {**calib, "met": not (0.9 <= calib["o_e"] <= 1.1)},
    "S1": diffs["S1-M2"],
    "thresholds": {"T1_f1_cutoff_on_M2": t1},
}
(RES / "predictions.json").write_text(json.dumps(verdicts, indent=2, default=float))
print(ranking.round(4).to_string(index=False))
print(json.dumps(verdicts, indent=1, default=float))
