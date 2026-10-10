from pathlib import Path

import pandas as pd

from simulator.sensor_simulator import generate_sensor_data
from simulator.validate_dataset import validate_dataset


OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "generated"
)

NUM_SAMPLES = 1000
BASE_SEED = 42
NUM_SCENARIOS_PER_TYPE = 5

FAULT_TYPES = [
    "normal",
    "motor_overheating",
    "battery_degradation",
    "motor_imbalance",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for fault_index, fault_type in enumerate(FAULT_TYPES):
        for scenario in range(NUM_SCENARIOS_PER_TYPE):
            seed = BASE_SEED + fault_index * 100 + scenario

            # Normal datasets have no injected fault.
            if fault_type == "normal":
                selected_fault = None
                fault_start = 0
                fault_duration = 1
            else:
                selected_fault = fault_type

                # Vary fault timing and duration between scenarios.
                fault_start = 200 + scenario * 100
                fault_duration = 50 + scenario * 25

            print(
                f"\nGenerating {fault_type}, "
                f"scenario {scenario + 1}/{NUM_SCENARIOS_PER_TYPE}"
            )

            df = generate_sensor_data(
                num_samples=NUM_SAMPLES,
                seed=seed,
                fault_type=selected_fault,
                fault_start=fault_start,
                fault_duration=fault_duration,
            )

            filename = (
                f"{fault_type}_scenario_{scenario + 1}.csv"
            )
            output_path = OUTPUT_DIR / filename
            df.to_csv(output_path, index=False)

            report = validate_dataset(output_path)

            print(f"Saved: {output_path}")
            print(f"Fault samples: {int(df['fault_label'].sum())}")
            print(f"Validation passed: {report['valid']}")

            if not report["valid"]:
                raise RuntimeError(
                    f"Validation failed for {filename}: "
                    f"{report['errors']}"
                )

    print("\nAll scenario datasets generated and validated.")


if __name__ == "__main__":
    main()
