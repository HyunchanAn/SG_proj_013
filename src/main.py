from fastapi import FastAPI
from src.schemas import ReverseEngineeringInput, VerificationResult
from src.predictor import verify_and_predict

app = FastAPI(title="SG_proj_013 - Reverse Engineering QA Gateway")

@app.post("/verify", response_model=VerificationResult)
def verify(data: ReverseEngineeringInput):
    """
    역설계 검증 및 물성 예측을 수행합니다.
    """
    return verify_and_predict(data)
