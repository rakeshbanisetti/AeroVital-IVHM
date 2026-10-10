from pathlib import Path

import pandas as pd

from simulator.validate_dataset import validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "generated"
OUTPUT_PATH = DATA_DIR / "combined_uav_dataset.csv"


def main() -> None:
    scenario_files = sorted(
        DATA_DIR.glob("*_scenario_*.csv")
    )

    if not scenario_files:
        raise FileNotFoundError(
            f"No scenario CSV files found in {DATA_DIR}"
        )

    frames = []

    for file_path in scenario_files:
        df = pd.read_csv(file_path)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="raise",
        )

        # Verify timestamp order within each source scenario.
        if not df["timestamp"].is_monotonic_increasing:
            raise ValueError(
                f"Timestamps are not ordered in {file_path.name}"
            )

        df["scenario_source"] = file_path.stem
        frames.append(df)

        print(f"Loaded: {file_path.name} ({len(df)} rows)")

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    # Save a single combined file. Scenario boundaries are preserved.
    combined.to_csv(OUTPUT_PATH, index=False)

    print("\nCombined dataset summary")
    print("=" * 32)
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows: {len(combined)}")
    print(f"Columns: {len(combined.columns)}")

    print("\nFault-label distribution:")
    print(combined["fault_label"].value_counts().sort_index())

    print("\nFault-type distribution:")
    print(combined["fault_type"].value_counts())

    # Validate each scenario independently because each has its
    # own timestamp origin.
    print("\nScenario validation")
    all_valid = True

    for source, group in combined.groupby("scenario_source"):
        scenario_path = DATA_DIR / f"{source}.csv"
        report = validate_dataset(scenario_path)

        print(f"{source}: {report['valid']}")

        if not report["valid"]:
            all_valid = False
            print(f"  Errors: {report['errors']}")

    if not all_valid:
        raise RuntimeError(
            "One or more source scenarios failed validation."
        )

    print("\nAll source scenarios validated successfully.")
    print("Combined dataset created successfully.")


if __name__ == "__main__":
    main()
