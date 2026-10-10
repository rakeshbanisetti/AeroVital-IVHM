from app.services.health_assessment import assess_health


def test_model_disagreement_is_flagged_as_uncertain():
    result = assess_health(
        is_anomaly=True,
        predicted_fault="normal",
        diagnosis_status="model_disagreement",
    )

    assert result["health_score"] == 50
    assert result["risk_level"] == "uncertain"
    assert result["maintenance_recommendation"]


def test_consistent_fault_prediction_is_high_risk():
    result = assess_health(
        is_anomaly=True,
        predicted_fault="motor_overheating",
        diagnosis_status="consistent",
    )

    assert result["health_score"] == 30
    assert result["risk_level"] == "high"
    assert "motor overheating" in result["maintenance_recommendation"]