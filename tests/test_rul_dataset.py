import pandas as pd

from ml.generate_rul_dataset import generate_rul_dataset


def test_generated_rul_dataset_has_expected_scenarios_and_failures():
    data = generate_rul_dataset(
        num_scenarios=3,
        samples_per_scenario=20,
    )

    assert len(data) == 60
    assert data["scenario_source"].nunique() == 3
    assert data["failure_event"].sum() == 3
    assert data["rul_seconds"].notna().all()


def test_each_scenario_has_rul_countdown_to_zero():
    data = generate_rul_dataset(
        num_scenarios=2,
        samples_per_scenario=10,
    )

    for _, scenario in data.groupby("scenario_source"):
        scenario = scenario.sort_values("timestamp")

        assert scenario["rul_seconds"].iloc[0] == 9
        assert scenario["rul_seconds"].iloc[-1] == 0
        assert scenario["failure_event"].sum() == 1
