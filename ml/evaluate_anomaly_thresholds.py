from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "ml" / "artifacts"

FEATURES = joblib.load(ARTIFACT_DIR / "anomaly_features.joblib")
model = joblib.load(ARTIFACT_DIR / "isolation_forest.joblib")
validation = pd.read_csv(DATA_DIR / "validation.csv")

X = validation[FEATURES]
y = validation["fault_label"].astype(int)

# Higher anomaly scores indicate more unusual readings.
scores = -model.decision_function(X)

print("Anomaly score ranges:")
print(validation.assign(anomaly_score=scores).groupby("fault_label")["anomaly_score"].describe())

print("\nThreshold comparison (validation data only):")
for threshold in sorted(set(pd.Series(scores).quantile([0.90, 0.92, 0.94, 0.96, 0.98]).round(6))):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()

    print(
        f"threshold={threshold:.6f} | "
        f"precision={precision_score(y, predictions, zero_division=0):.3f} | "
        f"recall={recall_score(y, predictions, zero_division=0):.3f} | "
        f"F1={f1_score(y, predictions, zero_division=0):.3f} | "
        f"false_alarms={fp} | missed_faults={fn}"
    )
