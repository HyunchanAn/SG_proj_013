from typing import Any

from pydantic import BaseModel, Field


class ReverseEngineeringInput(BaseModel):
    target_properties: dict[str, float]
    xgboost_prediction: dict[str, float]
    ir_gnn_features: list[float]
    current_iteration: int = Field(default=1, ge=1, le=5, description="무한 루프 방지")
    
class VerificationResult(BaseModel):
    is_passed: bool
    predicted_properties: dict[str, float]
    error_rates: dict[str, float]
    confidence_score: float
    feedback_signal: dict[str, Any] | None = Field(None, description="오차 초과 시 보정 파라미터 제안")
