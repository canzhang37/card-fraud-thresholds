"""Join the tables, build history features."""
import duckdb

TRAIN_END, VALID_END = 121, 151  # last day of training and of validation

UID = """concat_ws('|', coalesce(CAST(card1 AS VARCHAR), 'NA'),
                       coalesce(CAST(addr1 AS VARCHAR), 'NA'),
                       coalesce(P_emaildomain, 'NA'))"""

# frames end 1 s back: same-second rows unseen
HISTORY_SQL = f"""
WITH base AS (
    SELECT TransactionID, TransactionDT, TransactionAmt, card1,
           coalesce(CAST(addr1 AS VARCHAR), 'NA') AS addr_key,
           coalesce(P_emaildomain, 'NA') AS email_key,
           {UID} AS uid
    FROM {{src}}
)
SELECT TransactionID, uid,
       count(*) OVER (PARTITION BY uid ORDER BY TransactionDT
                      RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING) AS uid_n_1h,
       count(*) OVER (PARTITION BY uid ORDER BY TransactionDT
                      RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING) AS uid_n_24h,
       count(*) OVER (PARTITION BY uid ORDER BY TransactionDT
                      RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING) AS uid_n_7d,
       TransactionDT - max(TransactionDT) OVER (PARTITION BY uid ORDER BY TransactionDT
                      RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS uid_secs_since_prev,
       TransactionAmt / avg(TransactionAmt) OVER (PARTITION BY uid ORDER BY TransactionDT
                      RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS uid_amt_ratio,
       count(DISTINCT card1) OVER (PARTITION BY email_key, addr_key ORDER BY TransactionDT
                      RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING) AS email_addr_cards_7d
FROM base
"""

HISTORY_COLS = ["uid_n_1h", "uid_n_24h", "uid_n_7d", "uid_secs_since_prev",
                "uid_amt_ratio", "email_addr_cards_7d"]


def history_features(con: duckdb.DuckDBPyConnection, src: str):
    """History features for each row of `src`."""
    return con.sql(HISTORY_SQL.format(src=src))


def build(con: duckdb.DuckDBPyConnection, tx_csv: str, id_csv: str) -> None:
    """Build table `joined` with splits and features."""
    con.sql(f"CREATE OR REPLACE TABLE tx AS SELECT * FROM read_csv_auto('{tx_csv}', sample_size=-1)")
    con.sql(f"CREATE OR REPLACE TABLE ident AS SELECT * FROM read_csv_auto('{id_csv}', sample_size=-1)")
    con.sql("CREATE OR REPLACE TABLE hist AS " + HISTORY_SQL.format(src="tx"))
    con.sql(f"""
        CREATE OR REPLACE TABLE joined AS
        SELECT t.*, i.* EXCLUDE (TransactionID),
               i.TransactionID IS NOT NULL AS has_identity,
               t.TransactionDT // 86400 AS day,
               CASE WHEN t.TransactionDT // 86400 <= {TRAIN_END} THEN 'train'
                    WHEN t.TransactionDT // 86400 <= {VALID_END} THEN 'valid'
                    ELSE 'test' END AS split,
               h.* EXCLUDE (TransactionID)
        FROM tx t
        LEFT JOIN ident i USING (TransactionID)
        JOIN hist h USING (TransactionID)
        ORDER BY t.TransactionDT, t.TransactionID
    """)
