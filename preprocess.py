import pandas as pd
import joblib

ARTIFACTS = joblib.load("model_artifacts.joblib")

def preprocess(raw_input: dict) -> pd.DataFrame:
    """
    raw_input keys required (raw, unencoded values):
    checking_balance, months_loan_duration, credit_history, purpose, amount,
    savings_balance, employment_duration, percent_of_income, years_at_residence,
    age, other_credit, housing, existing_loans_count, job, dependents, phone
    """
    df = pd.DataFrame([raw_input])

    nominal_cols = ARTIFACTS['onehot_encoder_cols']
    ordinal_cols = ARTIFACTS['ordinal_cols']
    encoder = ARTIFACTS['onehot_encoder']
    ordinal_encoder = ARTIFACTS['ordinal_encoder']
    scaler = ARTIFACTS['scaler']
    feature_columns = ARTIFACTS['feature_columns']

    encoded_nominal = encoder.transform(df[nominal_cols])
    df_nominal_encoded = pd.DataFrame(
        encoded_nominal.toarray(),
        columns=encoder.get_feature_names_out(input_features=nominal_cols)
    )

    df_ordinal_encoded = pd.DataFrame(
        ordinal_encoder.transform(df[ordinal_cols]),
        columns=[f"{col}_encoded" for col in ordinal_cols]
    )

    df_final = pd.concat(
        [df.drop(columns=nominal_cols + ordinal_cols), df_nominal_encoded, df_ordinal_encoded],
        axis=1
    )

    df_final = df_final[feature_columns]

    scaled = scaler.transform(df_final)
    return scaled