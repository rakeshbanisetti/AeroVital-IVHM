from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "ml" / "artifacts"
TEST_PATH = DATA_DIR / "test.csv"


def main():
    test_df = pd.read_csv(TEST_PATH)

    # 1. Evaluate fault diagnosis.
    diagnosis = joblib.load(ARTIFACT_DIR / "fault_diagnosis_rf.joblib")
    diagnosis_features = joblib.load(
        ARTIFACT_DIR / "diagnosis_features.joblib"
    )

    X_diagnosis = test_df[diagnosis_features]
    y_diagnosis = test_df["fault_type"]
    diagnosis_predictions = diagnosis.predict(X_diagnosis)

    print("\n=== FAULT DIAGNOSIS: HELD-OUT TEST SET ===")
    print(f"Test samples: {len(test_df)}")
    print(f"Accuracy: {accuracy_score(y_diagnosis, diagnosis_predictions):.4f}")
    print(
        "Macro F1: "
        f"{f1_score(y_diagnosis, diagnosis_predictions, average='macro'):.4f}"
    )
    print("\nClass order:", list(diagnosis.classes_))
    print(
        confusion_matrix(
            y_diagnosis,
            diagnosis_predictions,
            labels=diagnosis.classes_,
        )
    )
    print("\nClassification report:")
    print(
        classification_report(
            y_diagnosis,
            diagnosis_predictions,
            labels=diagnosis.classes_,
            zero_division=0,
        )
    )

    # 2. Evaluate the Isolation Forest using its existing default threshold.
    anomaly_model = joblib.load(
        ARTIFACT_DIR / "isolation_forest.joblib"
    )
    anomaly_features = joblib.load(
        ARTIFACT_DIR / "anomaly_features.joblib"
    )

    X_anomaly = test_df[anomaly_features]
    y_anomaly = test_df["fault_label"].astype(int)
    anomaly_predictions = (
        anomaly_model.predict(X_anomaly) == -1
    ).astype(int)

    print("\n=== ANOMALY DETECTION: HELD-OUT TEST SET ===")
    print(f"Test samples: {len(test_df)}")
    print(
        "Precision: "
        f"{precision_score(y_anomaly, anomaly_predictions, zero_division=0):.4f}"
    )
    print(
        "Recall: "
        f"{recall_score(y_anomaly, anomaly_predictions, zero_division=0):.4f}"
    )
    print(
        "F1 score: "
        f"{f1_score(y_anomaly, anomaly_predictions, zero_division=0):.4f}"
    )
    print("\nConfusion matrix (rows=true, columns=predicted; labels 0, 1):")
    print(
        confusion_matrix(
            y_anomaly,
            anomaly_predictions,
            labels=[0, 1],
        )
    )

    print("\nEvaluation completed. No model was fitted or tuned in this script.")
    print("Results apply to the current synthetic test dataset.")


if __name__ == "__main__":
    main()
