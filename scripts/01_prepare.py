"""Join, add history features and splits, check V2-V3."""
import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraudcost import features  # noqa: E402

RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "interim"
OUT.mkdir(parents=True, exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)

t0 = time.time()
con = duckdb.connect()
features.build(con, (RAW / "train_transaction.csv").as_posix(), (RAW / "train_identity.csv").as_posix())

n_rows = con.sql("SELECT count(*) FROM joined").fetchone()[0]
assert n_rows == 590_540, n_rows

# V2: first transaction has no history
bad = con.sql("""
    SELECT count(*) FROM (
        SELECT *, row_number() OVER (PARTITION BY uid ORDER BY TransactionDT, TransactionID) AS k
        FROM joined) WHERE k = 1 AND (uid_n_7d > 0 OR uid_secs_since_prev IS NOT NULL)
""").fetchone()[0]
assert bad == 0, bad

# V3: every transaction is in exactly one split.
dup = con.sql("SELECT count(*) - count(DISTINCT TransactionID) FROM joined").fetchone()[0]
assert dup == 0

con.sql(f"COPY joined TO '{(OUT / 'full.parquet').as_posix()}' (FORMAT parquet)")
summary = con.sql("""
    SELECT split, min(day) AS first_day, max(day) AS last_day, count(*) AS transactions,
           sum(isFraud) AS frauds, round(avg(isFraud), 4) AS fraud_rate,
           count(DISTINCT uid) AS customers, round(sum(TransactionAmt)) AS dollars,
           round(sum(TransactionAmt * isFraud)) AS fraud_dollars
    FROM joined GROUP BY split ORDER BY first_day
""").df()
summary.to_csv(ROOT / "results" / "split_summary.csv", index=False)
print(summary.to_string(index=False))
print(f"V2 and V3 passed, {time.time() - t0:.0f} s")
