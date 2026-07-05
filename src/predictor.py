from .schemas import ReverseEngineeringInput, VerificationResult
from loguru import logger

def verify_and_predict(data: ReverseEngineeringInput) -> VerificationResult:
    """
    XGBoost 예측값과 IR GNN 특징을 앙상블하여 최종 물성을 검증 및 예측.
    최대 5회의 이터레이션을 통해 무한 루프 방지.
    """
    if data.current_iteration > 5:
        logger.warning(f"013: Max iteration limit exceeded ({data.current_iteration}). Stopping feedback loop.")
        return VerificationResult(
            is_passed=False,
            predicted_properties={},
            error_rates={},
            confidence_score=0.0,
            feedback_signal={"error": "Max iteration limit exceeded. Stopping feedback loop."}
        )

    # Dummy ensemble logic
    predicted_properties = data.xgboost_prediction.copy()
    error_rates = {}
    is_passed = True
    feedback_signal = None
    
    # GNN 특징 결합을 통한 물리 보정 레이어
    for key, target_val in data.target_properties.items():
        pred_val = predicted_properties.get(key, 0.0)
        
        if data.ir_gnn_features:
            # Analyze GNN structural embedding magnitude
            gnn_magnitude = sum(abs(v) for v in data.ir_gnn_features) / len(data.ir_gnn_features)
            
            # Apply dynamic chemical correction based on property type
            if "Tg" in key:
                # Polar intermolecular force shifts Tg upwards
                pred_val += gnn_magnitude * 15.0
            elif "측정_값" in key:
                # Adhesion correction scales with structural embedding polar density
                pred_val += (gnn_magnitude - 0.5) * 50.0
            else:
                pred_val += gnn_magnitude * 2.0

        predicted_properties[key] = pred_val
        
        if target_val != 0:
            error = abs(target_val - pred_val) / target_val
        else:
            error = abs(target_val - pred_val)
        
        error_rates[key] = error
        
        # Adaptive tolerance based on property classification
        tolerance = 0.10  # Default 10%
        if "Tg" in key:
            tolerance = 0.15  # Allow up to 15% deviation for thermal properties
        elif "측정_값" in key:
            tolerance = 0.12  # Allow up to 12% deviation for raw adhesion
            
        if error > tolerance:
            is_passed = False
            
    confidence_score = max(0.0, 1.0 - sum(error_rates.values()) / max(1, len(error_rates)))
    
    if not is_passed:
        logger.info(f"013: Iteration {data.current_iteration} failed. Confidence: {confidence_score:.3f}. Generating delta correction for next iteration.")
        feedback_signal = {
            "suggested_action": "adjust_parameters",
            "next_iteration": data.current_iteration + 1
        }
    else:
        logger.info(f"013: Iteration {data.current_iteration} passed! Confidence: {confidence_score:.3f}")
        
    return VerificationResult(
        is_passed=is_passed,
        predicted_properties=predicted_properties,
        error_rates=error_rates,
        confidence_score=confidence_score,
        feedback_signal=feedback_signal
    )
