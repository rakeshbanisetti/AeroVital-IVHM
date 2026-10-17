from pydantic import BaseModel, ConfigDict, Field


class SensorPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    motor_temp_c: float = Field(ge=-50, le=250)
    motor_vibration: float = Field(ge=0, le=100)
    motor_current_a: float = Field(ge=0, le=500)
    motor_rpm: float = Field(ge=0, le=100000)

    battery_voltage_v: float = Field(ge=0, le=200)
    battery_current_a: float = Field(ge=0, le=500)
    battery_temp_c: float = Field(ge=-50, le=150)
    battery_soc_pct: float = Field(ge=0, le=100)

    altitude_m: float = Field(ge=-1000, le=50000)
    speed_mps: float = Field(ge=0, le=1000)
    acceleration_mps2: float = Field(ge=-200, le=200)

    ambient_temp_c: float = Field(ge=-100, le=100)
    pressure_hpa: float = Field(gt=0, le=1200)
    humidity_pct: float = Field(ge=0, le=100)
    degradation_level: float = Field(ge=0, le=1)


class SensorPredictionResponse(BaseModel):
    is_anomaly: bool
    predicted_fault: str
    diagnosis_status: str
    predicted_rul_seconds: float
    health_score: int
    risk_level: str
    maintenance_recommendation: str
