"""Question 1: M1 under random and grouped 5-fold CV.

Usage: python scripts/03_cv.py random|grouped
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraudcost import metrics, models  # noqa: E402

kind = sys.argv[1]
t0 = time.time()
df = models.load(ROOT / "data" / "interim" / "full.parquet")
df = df[df.split != "test"].reset_index(drop=True)
cols = models.raw_cols(df)
rounds = json.loads((ROOT / "results" / "raw" / "fit_M1.json").read_text())["best_iteration"] + 1

if kind == "random":
    folds = StratifiedKFold(5, shuffle=True, random_state=models.SEED).split(df, df["isFraud"])
else:
    folds = GroupKFold(5, shuffle=True, random_state=models.SEED).split(df, df["isFraud"], df["uid"])

oof = np.full(len(df), np.nan)
for k, (fit_idx, out_idx) in enumerate(folds):
    score, _ = models.fit_xgb(df.iloc[fit_idx], None, df.iloc[out_idx], cols,
                              n_estimators=rounds, early_stop=False)
    oof[out_idx] = score
    print(f"fold {k + 1} done, {time.time() - t0:.0f} s", flush=True)

res = {"split": kind, "rounds": rounds, "rows": len(df),
       "ap": metrics.average_precision(df["isFraud"].to_numpy(), oof),
       "seconds": round(time.time() - t0)}
(ROOT / "results" / "raw" / f"cv_{kind}.json").write_text(json.dumps(res))
print(res)
