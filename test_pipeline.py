import pytest
from preprocess import preprocess, ARTIFACTS
from predict import predict

SAMPLE_LOW_RISK = {
    "checking_balance": "< 0 DM", "months_loan_duration": 6, "credit_history": "critical",
    "purpose": "furniture/appliances", "amount": 1169, "savings_balance": "unknown",
    "employment_duration": "> 7 years", "percent_of_income": 4, "years_at_residence": 4,
    "age": 67, "other_credit": "none", "housing": "own", "existing_loans_count": 2,
    "job": "skilled", "dependents": 1, "phone": "yes"
}

def test_preprocess_output_shape():
    result = preprocess(SAMPLE_LOW_RISK)
    assert result.shape == (1, 35)

def test_preprocess_returns_numeric():
    result = preprocess(SAMPLE_LOW_RISK)
    assert result.dtype.kind in ('f', 'i')

def test_predict_returns_expected_keys():
    result = predict(SAMPLE_LOW_RISK)
    assert set(result.keys()) == {"default_probability", "prediction", "threshold_used"}

def test_predict_probability_in_valid_range():
    result = predict(SAMPLE_LOW_RISK)
    assert 0.0 <= result["default_probability"] <= 1.0

def test_predict_prediction_is_valid_label():
    result = predict(SAMPLE_LOW_RISK)
    assert result["prediction"] in ("yes", "no")

def test_predict_threshold_matches_artifact():
    result = predict(SAMPLE_LOW_RISK)
    assert result["threshold_used"] == ARTIFACTS["threshold"]

def test_predict_consistent_across_calls():
    result1 = predict(SAMPLE_LOW_RISK)
    result2 = predict(SAMPLE_LOW_RISK)
    assert result1 == result2