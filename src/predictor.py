from .schemas import ReverseEngineeringInput, VerificationResult

def verify_and_predict(data: ReverseEngineeringInput) -> VerificationResult:
    """
    XGBoost 예측값과 IR GNN 특징을 앙상블하여 최종 물성을 검증 및 예측.
    최대 5회의 이터레이션을 통해 무한 루프 방지.
    """
    if data.current_iteration > 5:
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
    
    # Simple evaluation
    for key, target_val in data.target_properties.items():
        pred_val = predicted_properties.get(key, 0.0)
        # Apply some IR GNN dummy feature effect
        if data.ir_gnn_features:
            pred_val += sum(data.ir_gnn_features) * 0.01

        predicted_properties[key] = pred_val
        
        if target_val != 0:
            error = abs(target_val - pred_val) / target_val
        else:
            error = abs(target_val - pred_val)
        
        error_rates[key] = error
        if error > 0.1:  # 10% error threshold
            is_passed = False
            
    confidence_score = max(0.0, 1.0 - sum(error_rates.values()) / max(1, len(error_rates)))
    
    if not is_passed:
        feedback_signal = {
            "suggested_action": "adjust_parameters",
            "next_iteration": data.current_iteration + 1
        }
        
    return VerificationResult(
        is_passed=is_passed,
        predicted_properties=predicted_properties,
        error_rates=error_rates,
        confidence_score=confidence_score,
        feedback_signal=feedback_signal
    )
