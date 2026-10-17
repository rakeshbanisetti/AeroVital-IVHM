from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "generated" / "rul_dataset.csv"
ARTIFACT_DIR = ROOT / "ml" / "artifacts"
REPORT_PATH = ROOT / "ml" / "RUL_BASELINE.md"

FEATURES = [
    "motor_temp_c",
    "motor_vibration",
    "motor_current_a",
    "motor_rpm",
    "degradation_level",
]

TARGET = "rul_seconds"


def main():
    data = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET, "scenario_source"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")

    if data[required].isna().any().any():
        raise ValueError("Dataset contains missing values in required columns.")

    # Split by scenario to prevent rows from the same scenario
    # appearing in both training and evaluation sets.
    scenarios = sorted(data["scenario_source"].unique())
    if len(scenarios) < 5:
        raise ValueError("At least five scenarios are required.")

    test_scenarios = set(scenarios[::5])
    test_mask = data["scenario_source"].isin(test_scenarios)

    train_data = data.loc[~test_mask]
    test_data = data.loc[test_mask]

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_data[FEATURES], train_data[TARGET])

    predictions = model.predict(test_data[FEATURES])
    mae = mean_absolute_error(test_data[TARGET], predictions)
    rmse = mean_squared_error(test_data[TARGET], predictions) ** 0.5
    r2 = r2_score(test_data[TARGET], predictions)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "rul_random_forest.joblib")
    joblib.dump(FEATURES, ARTIFACT_DIR / "rul_features.joblib")

    report = f"""# RUL Regression Baseline

## Dataset and split
- Dataset: synthetic motor degradation scenarios
- Total rows: {len(data)}
- Total scenarios: {len(scenarios)}
- Training rows: {len(train_data)}
- Test rows: {len(test_data)}
- Held-out test scenarios: {len(test_scenarios)}
- Features: {", ".join(FEATURES)}
- Target: `{TARGET}` in seconds

## Held-out scenario metrics
- Mean Absolute Error (MAE): {mae:.3f} seconds
- Root Mean Squared Error (RMSE): {rmse:.3f} seconds
- R-squared: {r2:.4f}

## Limitations
This is an initial baseline evaluated on synthetic scenarios. The
degradation patterns and failure times are simulated, not validated
against real UAV operating or maintenance data. These metrics do not
establish real-world remaining-useful-life accuracy.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Training rows: {len(train_data):,}")
    print(f"Test rows: {len(test_data):,}")
    print(f"Held-out scenarios: {len(test_scenarios)}")
    print(f"MAE: {mae:.3f} seconds")
    print(f"RMSE: {rmse:.3f} seconds")
    print(f"R²: {r2:.4f}")
    print(f"Model saved: {ARTIFACT_DIR / 'rul_random_forest.joblib'}")
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
