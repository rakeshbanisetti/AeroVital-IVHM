def assess_health(is_anomaly: bool, predicted_fault: str, diagnosis_status: str) -> dict:
    if diagnosis_status == "model_disagreement":
        health_score = 50
        risk_level = "uncertain"
        recommendation = "Collect additional sensor readings and inspect the UAV."
    elif is_anomaly and predicted_fault != "normal":
        health_score = 30
        risk_level = "high"
        recommendation = f"Inspect the UAV for {predicted_fault.replace('_', ' ')}."
    elif is_anomaly:
        health_score = 50
        risk_level = "medium"
        recommendation = "Review sensor readings and run additional diagnostics."
    else:
        health_score = 95
        risk_level = "low"
        recommendation = "Continue routine monitoring and scheduled maintenance."

    return {
        "health_score": health_score,
        "risk_level": risk_level,
        "maintenance_recommendation": recommendation,
    }