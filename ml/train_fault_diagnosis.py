from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "ml" / "artifacts"

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

TARGET = "fault_type"


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    validation_df = pd.read_csv(DATA_DIR / "validation.csv")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_validation = validation_df[FEATURES]
    y_validation = validation_df[TARGET]

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)

    print("\n=== AeroVital Fault Diagnosis: Validation Results ===")
    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Accuracy: {accuracy_score(y_validation, predictions):.4f}")
    print(
        "Macro F1: "
        f"{f1_score(y_validation, predictions, average='macro'):.4f}"
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(
        y_validation,
        predictions,
        labels=model.classes_,
    ))
    print("\nClass order:", list(model.classes_))
    print("\nClassification report:")
    print(classification_report(
        y_validation,
        predictions,
        zero_division=0,
    ))

    model_path = ARTIFACT_DIR / "fault_diagnosis_rf.joblib"
    features_path = ARTIFACT_DIR / "diagnosis_features.joblib"

    joblib.dump(model, model_path)
    joblib.dump(FEATURES, features_path)

    print(f"\nModel saved to: {model_path}")
    print(f"Feature list saved to: {features_path}")
    print("Reminder: results are based on synthetic data.")


if __name__ == "__main__":
    main()