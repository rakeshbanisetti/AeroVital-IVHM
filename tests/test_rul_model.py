from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "ml" / "artifacts"


def test_rul_model_artifacts_exist():
    assert (ARTIFACT_DIR / "rul_random_forest.joblib").is_file()
    assert (ARTIFACT_DIR / "rul_features.joblib").is_file()


def test_rul_model_predicts_finite_nonnegative_values():
    model = joblib.load(ARTIFACT_DIR / "rul_random_forest.joblib")
    features = joblib.load(ARTIFACT_DIR / "rul_features.joblib")

    sample = pd.DataFrame(
        [
            {
                "motor_temp_c": 50.0,
                "motor_vibration": 0.4,
                "motor_current_a": 6.0,
                "motor_rpm": 4200.0,
                "degradation_level": 0.5,
            }
        ],
        columns=features,
    )

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert pd.notna(prediction[0])
    assert prediction[0] >= 0
