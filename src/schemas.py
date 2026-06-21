from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any

class ReverseEngineeringInput(BaseModel):
    target_properties: Dict[str, float]
    xgboost_prediction: Dict[str, float]
    ir_gnn_features: List[float]
    current_iteration: int = Field(default=1, ge=1, le=5, description="무한 루프 방지")
    
class VerificationResult(BaseModel):
    is_passed: bool
    predicted_properties: Dict[str, float]
    error_rates: Dict[str, float]
    confidence_score: float
    feedback_signal: Optional[Dict[str, Any]] = Field(None, description="오차 초과 시 보정 파라미터 제안")
