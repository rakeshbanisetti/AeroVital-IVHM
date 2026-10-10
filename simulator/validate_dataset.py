from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "flight_load",
    "ambient_temp_c",
    "motor_temp_c",
    "motor_vibration",
    "motor_current_a",
    "motor_rpm",
    "battery_voltage_v",
    "battery_current_a",
    "battery_temp_c",
    "battery_soc_pct",
    "electrical_power_w",
    "altitude_m",
    "speed_mps",
    "acceleration_mps2",
    "pressure_hpa",
    "humidity_pct",
    "fault_label",
    "fault_type",
}

NUMERIC_COLUMNS = REQUIRED_COLUMNS - {
    "timestamp",
    "fault_type",
}


def validate_dataset(
    csv_path: str | Path,
) -> dict:
    """Validate a generated UAV sensor dataset."""

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    df = pd.read_csv(path)

    errors = []
    warnings = []

    # Check required columns.
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        errors.append(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if df.empty:
        errors.append("Dataset contains no rows.")

    if errors:
        return {
            "valid": False,
            "rows": len(df),
            "columns": len(df.columns),
            "errors": errors,
            "warnings": warnings,
        }

    # Check missing values.
    missing_values = int(df.isna().sum().sum())

    if missing_values:
        errors.append(
            f"Found {missing_values} missing values."
        )

    # Check numeric data types and finite values.
    for column in sorted(NUMERIC_COLUMNS):
        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if converted.isna().any():
            errors.append(
                f"Column '{column}' contains "
                "non-numeric or missing values."
            )
        elif not converted.map(
            lambda value: abs(value) != float("inf")
        ).all():
            errors.append(
                f"Column '{column}' contains infinite values."
            )

    # Check timestamps.
    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    if timestamps.isna().any():
        errors.append(
            "Timestamp column contains invalid values."
        )
    else:
        if timestamps.duplicated().any():
            warnings.append(
                "Duplicate timestamps were found."
            )

        if not timestamps.is_monotonic_increasing:
            errors.append(
                "Timestamps are not in chronological order."
            )

    # Check fault labels.
    labels = set(df["fault_label"].unique())

    if not labels.issubset({0, 1}):
        errors.append(
            "fault_label must contain only 0 or 1."
        )

    allowed_fault_types = {
        "normal",
        "motor_overheating",
        "battery_degradation",
        "motor_imbalance",
    }

    observed_fault_types = set(df["fault_type"].dropna().unique())

    if not observed_fault_types.issubset(
        allowed_fault_types
    ):
        errors.append(
            "Dataset contains unsupported fault_type values."
        )

    if (
        "fault_label" in df.columns
        and "fault_type" in df.columns
    ):
        inconsistent = (
            ((df["fault_label"] == 0) & (df["fault_type"] != "normal"))
            | ((df["fault_label"] == 1) & (df["fault_type"] == "normal"))
        )

        if inconsistent.any():
            errors.append(
                "fault_label and fault_type are inconsistent."
            )

    # Check basic physical ranges.
    range_checks = {
        "flight_load": (0, 1),
        "motor_vibration": (0, None),
        "motor_current_a": (0, None),
        "motor_rpm": (0, None),
        "battery_soc_pct": (0, 100),
        "humidity_pct": (0, 100),
        "speed_mps": (0, None),
        "pressure_hpa": (0, None),
    }

    for column, (minimum, maximum) in range_checks.items():
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid = pd.Series(False, index=df.index)

        if minimum is not None:
            invalid |= values < minimum

        if maximum is not None:
            invalid |= values > maximum

        if invalid.any():
            warnings.append(
                f"Column '{column}' has values outside "
                "the expected range."
            )

    return {
        "valid": len(errors) == 0,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": missing_values,
        "fault_counts": {
            str(key): int(value)
            for key, value in df["fault_type"].value_counts().items()
        },
        "errors": errors,
        "warnings": warnings,
    }


if __name__ == "__main__":
    dataset_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "generated"
        / "uav_sensor_data.csv"
    )

    report = validate_dataset(dataset_path)

    print("AeroVital Dataset Validation")
    print("=" * 32)

    for key, value in report.items():
        print(f"{key}: {value}")

    if not report["valid"]:
        raise SystemExit("Dataset validation failed.")
