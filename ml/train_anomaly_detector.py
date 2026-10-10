from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"

FEATURES = [
    "motor_temp_c",
    "motor_vibration",
    "motor_current_a",
    "motor_rpm",
    "battery_voltage_v",
    "battery_current_a",
    "battery_temp_c",
    "battery_soc_pct",
    "altitude_m",
    "speed_mps",
    "acceleration_mps2",
    "ambient_temp_c",
    "pressure_hpa",
    "humidity_pct",
]

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    # Learn normal operating patterns only.
    normal_train = train_df.loc[train_df["fault_label"] == 0]

    X_train = normal_train[FEATURES]
    X_validation = validation_df[FEATURES]
    y_validation = validation_df["fault_label"].astype(int)

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    # Isolation Forest returns -1 for anomalies and +1 for normal readings.
    predictions = (model.predict(X_validation) == -1).astype(int)

    print("\n=== AeroVital Anomaly Detection: Validation Results ===")
    print(f"Normal training readings: {len(normal_train)}")
    print(f"Validation readings: {len(validation_df)}")
    print(f"Features used: {len(FEATURES)}")
    print(f"Precision: {precision_score(y_validation, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_validation, predictions, zero_division=0):.4f}")
    print(f"F1 score:  {f1_score(y_validation, predictions, zero_division=0):.4f}")
    print("\nConfusion matrix (rows=true, columns=predicted; labels 0, 1):")
    print(confusion_matrix(y_validation, predictions, labels=[0, 1]))
    print("\nClassification report:")
    print(
        classification_report(
            y_validation,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Fault"],
            zero_division=0,
        )
    )

    model_path = ARTIFACT_DIR / "isolation_forest.joblib"
    metadata_path = ARTIFACT_DIR / "anomaly_features.joblib"

    joblib.dump(model, model_path)
    joblib.dump(FEATURES, metadata_path)

    print(f"Model saved to: {model_path}")
    print(f"Feature list saved to: {metadata_path}")
    print("\nNote: these results use synthetic data and are not real-world UAV performance.")

if __name__ == "__main__":
    main()
