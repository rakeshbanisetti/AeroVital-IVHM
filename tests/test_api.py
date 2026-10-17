
import uuid

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_create_and_list_uav():
    unique_code = f"TEST-{uuid.uuid4().hex[:10]}"

    payload = {
        "uav_code": unique_code,
        "model_name": "Automated Test UAV",
        "manufacturer": "AeroVital Test",
    }

    create_response = client.post("/api/v1/uavs/", json=payload)

    assert create_response.status_code == 201

    created_uav = create_response.json()
    assert created_uav["uav_code"] == unique_code
    assert created_uav["model_name"] == "Automated Test UAV"

    list_response = client.get("/api/v1/uavs/")

    assert list_response.status_code == 200
    assert any(
        uav["uav_code"] == unique_code
        for uav in list_response.json()
    )


def test_duplicate_uav_code_is_rejected():
    unique_code = f"DUP-{uuid.uuid4().hex[:10]}"

    payload = {
        "uav_code": unique_code,
        "model_name": "Duplicate Test UAV",
        "manufacturer": None,
    }

    first_response = client.post("/api/v1/uavs/", json=payload)
    second_response = client.post("/api/v1/uavs/", json=payload)

    assert first_response.status_code == 201
    assert second_response.status_code == 409


def test_sensor_prediction_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    sensor_data = {
        "motor_temp_c": 45,
        "motor_vibration": 0.1,
        "motor_current_a": 5,
        "motor_rpm": 4500,
        "battery_voltage_v": 22.2,
        "battery_current_a": 5,
        "battery_temp_c": 30,
        "battery_soc_pct": 90,
        "altitude_m": 50,
        "speed_mps": 10,
        "acceleration_mps2": 0.2,
        "ambient_temp_c": 25,
        "pressure_hpa": 1013,
        "humidity_pct": 40,
    }

    response = client.post("/api/v1/predictions/", json=sensor_data)

    assert response.status_code == 200
    result = response.json()
    assert isinstance(result["is_anomaly"], bool)
    assert isinstance(result["predicted_fault"], str)
    assert result["diagnosis_status"] in {
        "consistent",
        "model_disagreement",
    }
    assert isinstance(result["health_score"], int)
    assert 0 <= result["health_score"] <= 100
    assert result["risk_level"] in {"low", "medium", "high", "uncertain"}
    assert isinstance(result["maintenance_recommendation"], str)
    assert len(result["maintenance_recommendation"]) > 0

def test_sensor_prediction_rejects_missing_fields():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    response = client.post(
        "/api/v1/predictions/",
        json={"motor_temp_c": 45},
    )

    assert response.status_code == 422
