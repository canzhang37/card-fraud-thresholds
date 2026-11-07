import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraudcost import features, metrics  # noqa: E402


def toy_history():
    rows = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5, 6, 7],
        "TransactionDT": [0, 100, 100, 4000, 4050, 90000, 700000],
        "TransactionAmt": [10.0, 30.0, 50.0, 20.0, 25.0, 40.0, 10.0],
        "card1": [1, 1, 1, 1, 2, 1, 1],
        "addr1": [5] * 7,
        "P_emaildomain": ["a.com"] * 7,
    })
    con = duckdb.connect()
    con.register("rows", rows)
    return features.history_features(con, "rows").df().set_index("TransactionID").sort_index()


def test_history_uses_only_earlier_rows():
    h = toy_history()
    # rows 2 and 3 share a timestamp
    assert h.loc[2, "uid_n_1h"] == 1 and h.loc[2, "uid_secs_since_prev"] == 100
    assert h.loc[3, "uid_n_1h"] == 1 and h.loc[3, "uid_secs_since_prev"] == 100
    assert h.loc[4, "uid_n_1h"] == 0 and h.loc[4, "uid_n_24h"] == 3
    assert np.isclose(h.loc[4, "uid_amt_ratio"], 20 / 30)
    assert h.loc[5, "uid_n_7d"] == 0 and pd.isna(h.loc[5, "uid_secs_since_prev"])
    assert h.loc[5, "email_addr_cards_7d"] == 1
    assert h.loc[6, "uid_n_24h"] == 1 and h.loc[6, "uid_n_7d"] == 4
    assert h.loc[6, "email_addr_cards_7d"] == 2
    assert h.loc[7, "uid_n_7d"] == 0 and h.loc[7, "uid_secs_since_prev"] == 610000


def test_average_precision_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(1)
    y = rng.random(2000) < 0.1
    s = np.round(rng.random(2000) + y * 0.3, 2)
    w = rng.integers(0, 3, 2000).astype(float)
    assert np.isclose(metrics.average_precision(y, s), average_precision_score(y, s))
    assert np.isclose(metrics.average_precision(y, s, w), average_precision_score(y, s, sample_weight=w))


def test_cost_threshold_beats_neighbours():
    rng = np.random.default_rng(2)
    y = (rng.random(5000) < 0.05).astype(float)
    s = rng.random(5000) + y * 0.5
    a = rng.lognormal(4, 1, 5000)
    t = metrics.cost_threshold(y, s, a, 10.0)
    best = metrics.cost(y, s >= t, a, 10.0)
    for q in np.quantile(s, [0.5, 0.8, 0.9, 0.95, 0.99]):
        assert best <= metrics.cost(y, s >= q, a, 10.0) + 1e-6


def test_cluster_weights_keep_groups_together():
    g = np.array(["a", "a", "b", "c", "c", "c"])
    w = metrics.cluster_weights(g, 50, 0)
    assert w.shape == (50, 6)
    assert np.all(w[:, 0] == w[:, 1]) and np.all(w[:, 3] == w[:, 5])
    assert np.all(w[:, 0] + w[:, 2] + w[:, 3] == 3)
