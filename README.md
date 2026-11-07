# Card fraud scores under time drift, and thresholds set by money

A fraud model is usually scored on a random split and then used with one probability cut-off. On 590,540 card-not-present transactions from the IEEE-CIS competition (Vesta, six months, 3.5% fraud), this project trains on the first four months, tunes and calibrates on the fifth, and tests once on the sixth. The plan and five predictions were written down before any model was fitted ([docs/design.md](docs/design.md)); the results are in [docs/readout.md](docs/readout.md).

What came out:

- **Random splits flatter the model.** The same XGBoost model scores an average precision of 0.84 under random 5-fold cross-validation and 0.52 on the following month. Holding out whole customers brings it to 0.62. Fraud labels spread across a customer's later transactions, so a random split lets the model recognise customers rather than fraud.
- **The decision rule mattered more than the model.** Flagging a transaction when calibrated fraud probability × amount is at least the $10 review cost lost $185,000 in the test month, 42% less than the F1-optimal cut-off ($319,000) and 25% less than the best single cut-off ($246,000). It flags fewer transactions but catches more fraud dollars. The ordering held at $5 and $25 per review.
- **Extra features did not help.** Customer-history features built with SQL window functions (counts in the last hour, day and week, time since the last purchase, amount against the customer's history) and an isolation-forest anomaly score left average precision unchanged or slightly lower.
- **Calibration held for a month.** Isotonic calibration fitted on month five predicted 3,182 frauds in month six; 3,114 occurred. Week by week the ratio ranged from 0.86 to 1.08.

Intervals come from 1,000 bootstrap resamples of customer groups. Two of the five predictions failed (history features, calibration drift) and are reported as such.

## Run it

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows path; bin/ elsewhere
# Download ieee-fraud-detection.zip from the Kaggle competition and unzip the two train_* files into data/raw/
.venv/Scripts/python scripts/01_prepare.py      # DuckDB join, history features, splits (about 1 minute)
for m in M0 M1 M2 M3 M4 S1; do .venv/Scripts/python scripts/02_fit.py $m; done   # about 10 minutes
.venv/Scripts/python scripts/03_cv.py random    # 5 minutes
.venv/Scripts/python scripts/03_cv.py grouped   # 11 minutes
.venv/Scripts/python scripts/04_evaluate.py
.venv/Scripts/python scripts/05_figures.py
.venv/Scripts/python -m pytest tests
```

Times are for a 20-thread desktop CPU.

## Data

IEEE-CIS Fraud Detection (Kaggle, 2019), data from Vesta Corporation. Kaggle's rules allow non-commercial academic research and education and forbid redistribution, so no data or row-level results are in this repository.

Can Zhang, zhan8639@umn.edu. MIT license.
