"""Feature sets and model fits for the time split."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import HISTORY_COLS

SEED = 2026
N_JOBS = 16
NOT_RAW = {"TransactionID", "TransactionDT", "isFraud", "has_identity", "day", "split", "uid",
           *HISTORY_COLS}
ID_COLS = [f"id_{k:02d}" for k in range(1, 39)] + ["DeviceType", "DeviceInfo"]
F0_CAT = ["ProductCD", "card4", "card6", "DeviceType"]
XGB_PARAMS = dict(tree_method="hist", max_depth=8, learning_rate=0.1, subsample=0.8,
                  colsample_bytree=0.5, enable_categorical=True, eval_metric="aucpr",
                  n_jobs=N_JOBS, random_state=SEED)


def load(path):
    df = pd.read_parquet(path)
    for c in df.columns:
        if c in ("split", "uid") or pd.api.types.is_numeric_dtype(df[c]):
            continue
        # text and T/F columns -> string categories
        s = df[c].astype(object)
        missing = s.isna()
        s = s.astype(str)
        s[missing] = np.nan
        df[c] = pd.Categorical(s)
    for c in HISTORY_COLS:
        df[c] = df[c].astype(float)
    df["has_identity"] = df["has_identity"].astype(float)
    return df


def raw_cols(df):
    return [c for c in df.columns if c not in NOT_RAW]


def f0_frame(df):
    out = pd.DataFrame({
        "log_amt": np.log1p(df["TransactionAmt"]),
        "hour": (df["TransactionDT"] // 3600) % 24,
        "email_match": (df["P_emaildomain"].astype(object) == df["R_emaildomain"].astype(object))
        .fillna(False).astype(float),
        "has_identity": df["has_identity"],
    }, index=df.index)
    for c in ID_COLS:
        out[f"miss_{c}"] = df[c].isna().astype(float)
    for c in F0_CAT:
        out[c] = df[c].astype(object).fillna("NA").astype(str)
    return out


def anomaly_frame(df):
    """Numeric F0 + F2 columns for the isolation forest."""
    out = f0_frame(df).drop(columns=F0_CAT)
    h = df[HISTORY_COLS].copy()
    h["uid_secs_since_prev"] = h["uid_secs_since_prev"].fillna(df["TransactionDT"].max())
    h["uid_amt_ratio"] = h["uid_amt_ratio"].fillna(1.0)
    return pd.concat([out, h], axis=1)


def fit_logistic(train, valid_test):
    num = [c for c in f0_frame(train.head(2)).columns if c not in F0_CAT]
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), F0_CAT)])
    model = make_pipeline(pre, LogisticRegression(C=1.0, max_iter=3000))
    model.fit(f0_frame(train), train["isFraud"])
    return model.predict_proba(f0_frame(valid_test))[:, 1], {}


def fit_xgb(train, valid, score_on, cols, n_estimators=1500, early_stop=True):
    params = dict(XGB_PARAMS, n_estimators=n_estimators)
    if early_stop:
        params["early_stopping_rounds"] = 100
    model = xgb.XGBClassifier(**params)
    model.fit(train[cols], train["isFraud"],
              eval_set=[(valid[cols], valid["isFraud"])] if early_stop else None, verbose=False)
    info = {"best_iteration": getattr(model, "best_iteration", n_estimators - 1),
            "n_features": len(cols)}
    return model.predict_proba(score_on[cols])[:, 1], info


def fit_isolation(train, score_on):
    model = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=N_JOBS)
    model.fit(anomaly_frame(train))
    return -model.score_samples(anomaly_frame(score_on))
