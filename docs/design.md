# Design: card fraud scores under time drift, and thresholds set by money rather than by F1

Status: frozen after check V1 below and before any model was fitted or any label was compared with a score. Later changes would go in amendments at the end; none were needed, and the changes made while running are listed under Limits in the readout.

## Question

A fraud model is usually reported with one number from a random split and then used with one probability cut-off. Payment fraud breaks both habits: fraud patterns move from week to week, labels arrive late, and a missed $2,000 fraud costs far more than a missed $20 one. On public card-not-present transactions this project asks:

1. How much does a random split flatter a fraud model compared with training on earlier months and testing on a later one?
2. Do features built from each customer's past transactions (counts, time since last purchase, amount against the customer's own history) improve a gradient-boosted model, and does an unsupervised anomaly score add anything once labels are used?
3. Does a decision rule that flags a transaction when its expected fraud loss exceeds the cost of acting on it, using calibrated probabilities, lose less money than the usual single cut-off?
4. How stable are performance and calibration from week to week after the model is fitted?

## Data

- IEEE-CIS Fraud Detection (Kaggle, 2019), card-not-present e-commerce transactions provided by Vesta. Only the labelled files are used: `train_transaction.csv` and `train_identity.csv`, joined on `TransactionID` (left join; not every transaction has an identity row). The two `test_*` files have no public labels and are not used.
- Use is limited by the competition rules to non-commercial academic research and education, and the data may not be redistributed. Nothing derived at the row level is committed; the repository holds code and aggregate results only.
- `TransactionDT` is seconds from an undisclosed reference time. Day = floor(TransactionDT / 86,400). The host does not give the calendar date, so weeks are counted from the first day in the file.
- Label definition, from the host's data description on the competition forum (discussion 101203): a transaction is labelled fraud when a chargeback is reported on the card, and later transactions linked to it by user account, e-mail address or billing address are labelled fraud too; a transaction is legitimate if nothing is reported within 120 days. Two consequences shape the design: fraud labels cluster within customers, and a label can depend on events after the transaction, so a model can learn the customer rather than the fraud when the same customer appears in training and test.
- There is no customer ID. A customer proxy is built as `uid = card1 | addr1 | P_emaildomain` (missing values kept as their own level). It is used for the history features and for clustered resampling, and its limits are stated in the readout.
- Check V1 (done before this plan was frozen): 590,540 transactions with 394 columns, 20,663 frauds (3.50%), 144,233 identity rows with 41 columns; `TransactionID` is unique in both tables and every identity row has a transaction. Days run from 1 to 182. Amounts run from $0.25 to $31,937 (median $68.77); fraud amounts total $3.08 million of $79.7 million. The zip from Kaggle is 124 MB and the five CSV files 1.35 GB.

## Splits

- Time split. Training = days 1-121, validation = days 122-151, test = days 152-182 (the first 4/6 of the 182 days, the next 1/6, the last 1/6). Validation is used for early stopping, calibration and choosing thresholds; test is touched once per model, at the end.
- Random split, for question 1 only: 5-fold stratified cross-validation on the training and validation months together.
- Customer-grouped split, for question 1 only: 5-fold cross-validation on the same rows with whole `uid` groups held out.
- The cross-validation fits use M1's settings with the number of boosting rounds fixed at the value M1 reached by early stopping in the time split, and their AP is computed on the pooled out-of-fold scores.

## Features

- F0 (small, for the logistic baseline): log(1 + amount), hour of day, `ProductCD`, `card4`, `card6`, `DeviceType`, whether `P_emaildomain` equals `R_emaildomain`, whether an identity row exists, and missing-value indicators for the identity fields.
- F1 (raw): every column of both tables except `TransactionID`, `TransactionDT` and the label, with string columns coded as categories.
- F2 (customer history, built in DuckDB SQL with window functions over strictly earlier transactions of the same `uid`): number of transactions in the previous 1 hour, 24 hours and 7 days; seconds since the previous transaction; amount divided by the mean of earlier amounts; number of distinct `card1` values seen with the same `P_emaildomain` and `addr1` in the previous 7 days. Earlier fraud labels are never used as features, because in production they would arrive weeks late.

## Models

Settings are fixed here; only the number of boosting rounds is chosen on validation.

- M0: logistic regression on F0 (one-hot categories, standardised numerics, L2 penalty with C = 1).
- M1: XGBoost on F1. Histogram trees, depth 8, learning rate 0.1, subsample 0.8, column sample 0.5, native handling of missing values and categories, up to 1,500 rounds with early stopping after 100 rounds on validation average precision.
- M2: M1 settings on F1 + F2.
- M3: isolation forest (200 trees, other settings at scikit-learn defaults) on the numeric columns of F0 and F2, fitted on the training months without labels; its anomaly score is used as a fraud score.
- M4: M2 with the M3 score added as a feature.

## Metrics

- Ranking: average precision (AP, primary), ROC AUC, recall at alert rates of 1%, 2% and 5%, and the share of fraud dollars caught at those alert rates.
- Calibration: probabilities from M2 are recalibrated by isotonic regression fitted on the validation month. Reported on test: Brier score, the observed-to-expected ratio of fraud counts (sum of labels / sum of calibrated probabilities), and a reliability table in ten bins.
- Money. For transaction i with amount a_i and an action cost c (manual review or step-up check, paid whenever a transaction is flagged): flagging costs c; not flagging a fraud costs a_i; not flagging a legitimate transaction costs 0. The base case is c = $10, with c = $5 and c = $25 as sensitivity analyses. This follows the example-dependent cost idea of Correa Bahnsen et al. (ICMLA 2013, Bayes minimum risk); the cost values are this project's assumptions, not theirs.
- Decision rules, each applied to M2 scores:
  - T1: one probability cut-off chosen to maximise F1 on validation.
  - T2: one probability cut-off chosen to minimise validation cost.
  - T3: Bayes minimum risk: flag when calibrated p_i × a_i ≥ c.
  Reported on test: total cost, cost per 1,000 transactions, alert rate, fraud dollars caught.
- Drift: weekly AP, weekly fraud rate and weekly observed-to-expected ratio across the validation and test months.
- Intervals: 95% intervals from 1,000 bootstrap resamples of `uid` groups in the test month. Model and policy comparisons use the same resamples (paired).

## Predictions

Each is stated with the rule that decides it.

- P1. Random-split optimism: for M1, the random 5-fold AP exceeds the time-split test AP by at least 0.05, and the customer-grouped AP lies between the two.
- P2. History features: M2 has higher test AP than M1; met if the paired interval of M2 − M1 lies above 0.
- P3. Anomaly score: M3 alone has less than half the test AP of M1, and adding it (M4 − M2) changes test AP by less than 0.01 with an interval that includes 0.
- P4. Money-based rule: under c = $10, T3 has lower test cost than T1; met if the paired interval of T3 − T1 lies below 0. Reported for T3 − T2 and for c = $5 and $25 without a pre-stated direction.
- P5. Calibration drift: the test-month observed-to-expected ratio of the validation-calibrated M2 lies outside 0.9 to 1.1.

## Sensitivity analysis (no prediction)

- S1. Label delay: refit M2 after dropping the last 30 days of the training months, as if chargebacks for those days had not arrived yet, and report the change in test AP and cost under T3.

## Checks

- V1. Row counts, fraud rate, key uniqueness and day range (done, see Data).
- V2. History features use only strictly earlier rows: a unit test on a toy table where the answer is known, and a check that the first transaction of every `uid` has zero prior counts.
- V3. No `TransactionID` appears in two of the training, validation and test sets.
- V4. The test month is scored once per model; any later change is reported as post hoc.

## Run plan

Python with DuckDB, pandas, scikit-learn and XGBoost on the laptop CPU. Expected run time is under 30 minutes in total (about ten XGBoost fits on about 450,000 rows), so no cluster job is needed.
