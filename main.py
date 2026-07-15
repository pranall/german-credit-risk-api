from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict

app = FastAPI(title="German Credit Risk API")

class LoanApplication(BaseModel):
    checking_balance: str
    months_loan_duration: int
    credit_history: str
    purpose: str
    amount: int
    savings_balance: str
    employment_duration: str
    percent_of_income: int
    years_at_residence: int
    age: int
    other_credit: str
    housing: str
    existing_loans_count: int
    job: str
    dependents: int
    phone: str

@app.get("/")
def root():
    return {"status": "German Credit Risk API is running"}

@app.post("/predict")
def get_prediction(application: LoanApplication):
    return predict(application.model_dump())