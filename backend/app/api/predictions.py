from fastapi import APIRouter, HTTPException
from app.services.health_assessment import assess_health

from app.schemas.prediction import (
    SensorPredictionRequest,
    SensorPredictionResponse,
)
from ml.prediction_service import prediction_service

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post("/", response_model=SensorPredictionResponse)
def predict_sensor_health(payload: SensorPredictionRequest):
    try:
        result = prediction_service.predict(payload.model_dump())
        health = assess_health(
            result["is_anomaly"],
            result["predicted_fault"],
            result["diagnosis_status"],
        )
        return {**result, **health}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
