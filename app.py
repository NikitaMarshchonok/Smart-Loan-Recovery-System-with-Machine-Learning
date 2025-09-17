from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
import math
import os
import json

app = FastAPI(title="Smart Loan Recovery – Mini API", version="0.1.0")

class BorrowerFeatures(BaseModel):
    Age: Optional[int] = Field(None, ge=18, le=100)
    Monthly_Income: float = Field(..., ge=0)
    Loan_Amount: float = Field(..., ge=0)
    Outstanding_Loan_Amount: float = Field(..., ge=0)
    Monthly_EMI: Optional[float] = Field(0, ge=0)
    Num_Missed_Payments: int = Field(..., ge=0)
    Days_Past_Due: int = Field(..., ge=0)
    Payment_History: Optional[str] = Field(None, description="One of: Good, Average, Poor")

class PredictionOut(BaseModel):
    risk_score: float
    strategy: str
    explanation: str

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOut)
def predict(features: BorrowerFeatures):
    """
    Rule-based baseline risk scoring.
    If you later train a model, replace this with a model.predict_proba pipeline.
    """
    # Base prior
    score = 0.50

    # Income pressure: outstanding to income ratio
    if features.Monthly_Income > 0:
        ratio = features.Outstanding_Loan_Amount / (features.Monthly_Income * 12.0 + 1e-9)
        score += 0.25 * clamp(ratio / 2.0)  # up to +0.25 when ratio >= 2 years of income

    # Missed payments & DPD
    score += 0.12 * clamp(features.Num_Missed_Payments / 6.0)   # up to +0.12 for >=6 misses
    score += 0.18 * clamp(features.Days_Past_Due / 90.0)        # up to +0.18 for >=90 dpd

    # Payment history category
    ph = (features.Payment_History or "").strip().lower()
    if ph == "good":
        score -= 0.20
    elif ph == "average":
        score += 0.00
    elif ph == "poor":
        score += 0.20

    score = clamp(score)

    # Strategy mapping
    if score > 0.75:
        strategy = "Immediate legal notices & aggressive recovery"
    elif score >= 0.50:
        strategy = "Settlement offers & repayment plans"
    else:
        strategy = "Automated reminders & monitoring"

    expl = (
        "Baseline 0.50 adjusted by debt-to-income, missed payments, days past due, "
        "and payment history. Replace with ML model when available."
    )
    return {"risk_score": round(score, 3), "strategy": strategy, "explanation": expl}
