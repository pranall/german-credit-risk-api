from predict import predict

sample = {
    "checking_balance": "< 0 DM", "months_loan_duration": 6, "credit_history": "critical",
    "purpose": "furniture/appliances", "amount": 1169, "savings_balance": "unknown",
    "employment_duration": "> 7 years", "percent_of_income": 4, "years_at_residence": 4,
    "age": 67, "other_credit": "none", "housing": "own", "existing_loans_count": 2,
    "job": "skilled", "dependents": 1, "phone": "yes"
}

result = predict(sample)
print(result)