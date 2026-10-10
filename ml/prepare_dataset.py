from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "generated"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

SCENARIO_PATTERN = "*_scenario_*.csv"
RANDOM_SEED = 42


def main() -> None:
    scenario_files = sorted(DATA_DIR.glob(SCENARIO_PATTERN))

    if len(scenario_files) < 5:
        raise ValueError(
            "At least five scenario files are required for splitting."
        )

    # Assign whole scenario files to one split only.
    scenario_names = [path.stem for path in scenario_files]

    scenario_info = []

    for path in scenario_files:
        df = pd.read_csv(path)

        if "fault_type" not in df.columns:
            raise ValueError(f"Missing fault_type in {path.name}")

        scenario_info.append(
            {
                "path": path,
                "name": path.stem,
                "category": (
                    "normal"
                    if df["fault_type"].nunique() == 1
                    and df["fault_type"].iloc[0] == "normal"
                    else path.stem.rsplit("_scenario_", 1)[0]
                ),
                "rows": len(df),
            }
        )

    # Split separately by category so each split contains all
    # available categories.
    rng = __import__("random").Random(RANDOM_SEED)
    assignments = {"train": [], "validation": [], "test": []}

    for category in sorted({item["category"] for item in scenario_info}):
        items = [
            item for item in scenario_info
            if item["category"] == category
        ]
        rng.shuffle(items)

        if len(items) < 3:
            raise ValueError(
                f"Category '{category}' needs at least three scenarios."
            )

        assignments["train"].extend(items[:-2])
        assignments["validation"].append(items[-2])
        assignments["test"].append(items[-1])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Confirm there is no scenario overlap between splits.
    split_names = {
        split: {item["name"] for item in items}
        for split, items in assignments.items()
    }

    assert split_names["train"].isdisjoint(split_names["validation"])
    assert split_names["train"].isdisjoint(split_names["test"])
    assert split_names["validation"].isdisjoint(split_names["test"])

    for split, items in assignments.items():
        frames = []

        for item in items:
            df = pd.read_csv(item["path"])
            df["scenario_source"] = item["name"]
            frames.append(df)

        result = pd.concat(frames, ignore_index=True)
        output_path = OUTPUT_DIR / f"{split}.csv"
        result.to_csv(output_path, index=False)

        print(f"\n{split.upper()} SET")
        print(f"Scenarios: {len(items)}")
        print(f"Rows: {len(result)}")
        print("Fault types:")
        print(result["fault_type"].value_counts().to_string())
        print(f"Saved: {output_path}")

    print("\nScenario-aware split completed.")
    print("No scenario appears in more than one split.")


if __name__ == "__main__":
    main()
