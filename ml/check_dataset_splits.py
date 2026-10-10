from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS = ["train", "validation", "test"]

REQUIRED_COLUMNS = {
    "timestamp",
    "scenario_source",
    "fault_label",
    "fault_type",
    "motor_temp_c",
    "motor_vibration",
    "battery_voltage_v",
}


def main() -> None:
    datasets = {}

    for split in SPLITS:
        path = DATA_DIR / f"{split}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing split: {path}")

        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise ValueError(
                f"{split} is missing columns: {sorted(missing)}"
            )

        if df.empty:
            raise ValueError(f"{split} is empty.")

        if df[list(REQUIRED_COLUMNS)].isna().any().any():
            raise ValueError(f"{split} has missing required values.")

        if not set(df["fault_label"].unique()).issubset({0, 1}):
            raise ValueError(f"{split} has invalid fault labels.")

        datasets[split] = df

        print(f"\n{split.upper()}")
        print(f"Rows: {len(df)}")
        print(f"Scenarios: {df['scenario_source'].nunique()}")
        print("Fault labels:")
        print(df["fault_label"].value_counts().sort_index().to_dict())

    scenario_sets = {
        split: set(df["scenario_source"].unique())
        for split, df in datasets.items()
    }

    assert scenario_sets["train"].isdisjoint(
        scenario_sets["validation"]
    )
    assert scenario_sets["train"].isdisjoint(
        scenario_sets["test"]
    )
    assert scenario_sets["validation"].isdisjoint(
        scenario_sets["test"]
    )

    print("\nPASS: required columns and labels are valid.")
    print("PASS: no scenario overlap between splits.")


if __name__ == "__main__":
    main()