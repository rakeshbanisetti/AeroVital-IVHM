
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
