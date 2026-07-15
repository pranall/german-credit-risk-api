from preprocess import preprocess, ARTIFACTS

def predict(raw_input: dict) -> dict:
    X_scaled = preprocess(raw_input)

    model = ARTIFACTS['model']
    threshold = ARTIFACTS['threshold']

    probability_default = model.predict_proba(X_scaled)[0, 1]
    prediction = int(probability_default >= threshold)

    return {
        "default_probability": round(float(probability_default), 4),
        "prediction": "yes" if prediction == 1 else "no",
        "threshold_used": threshold
    }