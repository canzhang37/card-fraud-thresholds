"""Fit one model, save validation and test scores.

Usage: python scripts/02_fit.py M0|M1|M2|M3|M4|S1
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraudcost import models  # noqa: E402
from fraudcost.features import HISTORY_COLS  # noqa: E402

name = sys.argv[1]
OUT = ROOT / "results" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
df = models.load(ROOT / "data" / "interim" / "full.parquet")
train, valid = df[df.split == "train"], df[df.split == "valid"]
scored = df[df.split != "train"]
raw = models.raw_cols(df)

if name == "M0":
    score, info = models.fit_logistic(train, scored)
elif name == "M1":
    score, info = models.fit_xgb(train, valid, scored, raw)
elif name in ("M2", "S1"):
    fit_rows = train if name == "M2" else train[train.day <= 91]  # S1: last 30 days unlabelled
    score, info = models.fit_xgb(fit_rows, valid, scored, raw + list(HISTORY_COLS))
elif name == "M3":
    score, info = models.fit_isolation(train, scored), {}
elif name == "M4":
    df["m3_score"] = models.fit_isolation(train, df)
    train, valid, scored = (df[df.split == "train"], df[df.split == "valid"], df[df.split != "train"])
    score, info = models.fit_xgb(train, valid, scored, raw + list(HISTORY_COLS) + ["m3_score"])
else:
    raise SystemExit(f"unknown model {name}")

pd.DataFrame({"TransactionID": scored["TransactionID"].to_numpy(), "score": score}).to_parquet(
    OUT / f"scores_{name}.parquet", index=False)
info.update(model=name, seconds=round(time.time() - t0), train_rows=int(
    (train if name != "S1" else train[train.day <= 91]).shape[0]))
(OUT / f"fit_{name}.json").write_text(json.dumps(info))
print(info)
