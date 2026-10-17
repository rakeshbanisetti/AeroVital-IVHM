
from pathlib import Path

import pandas as pd

from ml.rul_targets import add_rul_target
from simulator.rul_simulator import generate_degradation_data


def generate_rul_dataset(
    num_scenarios: int = 30,
    samples_per_scenario: int = 1000,
) -> pd.DataFrame:
    if num_scenarios < 1:
        raise ValueError("num_scenarios must be at least 1.")

    scenarios = []

    for scenario_id in range(num_scenarios):
        scenario = generate_degradation_data(
            num_samples=samples_per_scenario,
            seed=42 + scenario_id,
        )
        scenario["scenario_source"] = f"rul_scenario_{scenario_id + 1}"
        scenarios.append(scenario)

    combined = pd.concat(scenarios, ignore_index=True)
    return add_rul_target(combined)


if __name__ == "__main__":
    output_path = Path("data/generated/rul_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = generate_rul_dataset()
    dataset.to_csv(output_path, index=False)

    print(f"Saved {len(dataset):,} rows to {output_path}")
    print(f"Scenarios: {dataset['scenario_source'].nunique()}")
    print(f"Failure events: {dataset['failure_event'].sum()}")
    print(f"Missing RUL targets: {dataset['rul_seconds'].isna().sum()}")
