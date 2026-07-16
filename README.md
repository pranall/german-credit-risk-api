# German Credit Risk API

A deployed machine learning API that predicts loan default risk, based on the UCI German Credit dataset. Built end-to-end: data cleaning, model comparison across 8 algorithms, hyperparameter tuning, threshold calibration, containerization, and a REST API.

## Problem

Banks must decide whether to approve a loan application. Two error types matter differently:
- **False negative** (approving an applicant who defaults): direct financial loss.
- **False positive** (rejecting a good applicant): lost revenue, but no direct loss.

Because false negatives are costlier, the model is tuned to prioritize **recall** (catching defaulters) over raw accuracy, at a controlled cost to precision.

## Model

8 classifiers were compared (Logistic Regression, KNN, SVM, QDA, Random Forest, Gradient Boosting, AdaBoost, XGBoost) using AUC-PR (chosen over AUC-ROC due to class imbalance — 700 non-default vs 300 default cases). Gradient Boosting performed best (AUC-PR 0.628).

**Hyperparameters** (via GridSearchCV, 5-fold CV, F1-scored): `n_estimators=150`, `learning_rate=0.1`, `max_depth=3`, `subsample=0.7`, `min_samples_leaf=10`. The `subsample` and `min_samples_leaf` values are regularization controls added specifically to reduce overfitting (see Model Iteration below).

**Decision threshold:** 0.2481, selected via 5-fold cross-validated precision-recall curve on the training set only (not the test set), targeting ~75% recall on defaults.

## Performance (test set, held out, never used in training or threshold selection)

|                | Precision | Recall | F1   |
|----------------|-----------|--------|------|
| No Default (0) | 0.86      | 0.69   | 0.76 |
| Default (1)    | 0.50      | 0.73   | 0.59 |

Confusion matrix: TN=120, FP=55, FN=20, TP=55.

At this threshold, the model catches 73% of actual defaulters, at the cost of flagging 31% of good applicants as risky (55/175). This tradeoff reflects the business assumption that a missed default costs more than a false alarm.

## Model iteration: fixing overfitting

An earlier version of this model used unregularized Gradient Boosting with a threshold hand-picked directly from test-set metrics — a methodological error, since it let the test set leak into the threshold-selection process. That version showed 100% training recall but only 76% test recall (a 24-point generalization gap), and a train/test false-positive-rate gap of over 21 points — clear signs of overfitting and threshold leakage.

**Fix:** added regularization to the model (`subsample`, `min_samples_leaf`), switched GridSearchCV scoring from `recall` to `f1` (unconstrained recall optimization pushes toward predicting the positive class too often), and re-derived the threshold using 5-fold cross-validated predictions on the training set only. Result: train/test gaps roughly halved across all metrics, and the reported test recall (73%) is now within 2 points of what cross-validation predicted (75.1%) — confirming the threshold generalizes rather than being an artifact of one train/test split.

## API
POST /predict
Request body:
```json
{
  "checking_balance": "< 0 DM",
  "months_loan_duration": 6,
  "credit_history": "critical",
  "purpose": "furniture/appliances",
  "amount": 1169,
  "savings_balance": "unknown",
  "employment_duration": "> 7 years",
  "percent_of_income": 4,
  "years_at_residence": 4,
  "age": 67,
  "other_credit": "none",
  "housing": "own",
  "existing_loans_count": 2,
  "job": "skilled",
  "dependents": 1,
  "phone": "yes"
}
```

Response:
```json
{
  "default_probability": 0.0151,
  "prediction": "no",
  "threshold_used": 0.2481
}
```

## Run locally

docker build -t german-credit-api .
docker run -p 8000:8000 german-credit-api

Then visit `http://127.0.0.1:8000/docs`.

## Limitations

- Trained on 1,000 records from a 1994 German dataset — demographic and economic patterns may not transfer to other populations or eras.
- Test set is small (250 rows); confidence intervals on the reported metrics are wide.
- No monitoring, authentication, or logging — this is a portfolio deployment, not production-hardened.

## Stack

Python, scikit-learn, FastAPI, Docker, pandas, joblib