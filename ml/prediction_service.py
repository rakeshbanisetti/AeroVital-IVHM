from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "ml" / "artifacts"

ANOMALY_MODEL_PATH = ARTIFACT_DIR / "isolation_forest.joblib"
ANOMALY_FEATURES_PATH = ARTIFACT_DIR / "anomaly_features.joblib"

DIAGNOSIS_MODEL_PATH = ARTIFACT_DIR / "fault_diagnosis_rf.joblib"
DIAGNOSIS_FEATURES_PATH = ARTIFACT_DIR / "diagnosis_features.joblib"


class PredictionService:
    def __init__(self):
        self.anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
        self.anomaly_features = joblib.load(ANOMALY_FEATURES_PATH)

        self.diagnosis_model = joblib.load(DIAGNOSIS_MODEL_PATH)
        self.diagnosis_features = joblib.load(DIAGNOSIS_FEATURES_PATH)

    def predict(self, sensor_data: dict) -> dict:
        anomaly_input = pd.DataFrame(
            [sensor_data],
            columns=self.anomaly_features,
        )
        diagnosis_input = pd.DataFrame(
            [sensor_data],
            columns=self.diagnosis_features,
        )

        missing_features = [
            feature
            for feature in set(
                self.anomaly_features + self.diagnosis_features
            )
            if feature not in sensor_data
        ]

        if missing_features:
            raise ValueError(
                f"Missing required sensor features: {sorted(missing_features)}"
            )

        anomaly_prediction = int(
            self.anomaly_model.predict(anomaly_input)[0] == -1
        )

        diagnosis_prediction = str(
            self.diagnosis_model.predict(diagnosis_input)[0]
        )

        return {
            "is_anomaly": bool(anomaly_prediction),
            "predicted_fault": diagnosis_prediction,
            "diagnosis_status": (
                "consistent"
                if bool(anomaly_prediction) == (diagnosis_prediction != "normal")
                else "model_disagreement"
            ),
        }


prediction_service = PredictionService()
