import pandas as pd
import pytest
from ml.rul_targets import add_rul_target
from simulator.rul_simulator import generate_degradation_data


def test_rul_decreases_toward_failure():
    data = pd.DataFrame(
        {
            "scenario_source": ["scenario_1"] * 5,
            "timestamp": pd.date_range(
                "2026-01-01", periods=5, freq="s"
            ),
            "failure_event": [0, 0, 0, 0, 1],
        }
    )

    result = add_rul_target(data)

    assert result["rul_seconds"].tolist() == [4, 3, 2, 1, 0]


def test_rul_is_calculated_separately_for_each_scenario():
    data = pd.DataFrame(
        {
            "scenario_source": ["scenario_1"] * 3 + ["scenario_2"] * 2,
            "timestamp": list(pd.date_range("2026-01-01", periods=3, freq="s"))
            + list(pd.date_range("2026-01-02", periods=2, freq="s")),
            "failure_event": [0, 0, 1, 0, 1],
        }
    )

    result = add_rul_target(data)

    assert result["rul_seconds"].tolist() == [2, 1, 0, 1, 0]


def test_rul_rejects_missing_failure_event_column():
    data = pd.DataFrame(
        {
            "scenario_source": ["scenario_1"],
            "timestamp": ["2026-01-01T00:00:00Z"],
        }
    )

    with pytest.raises(ValueError, match="failure_event"):
        add_rul_target(data)


def test_rul_rejects_invalid_failure_event_value():
    data = pd.DataFrame(
        {
            "scenario_source": ["scenario_1", "scenario_1"],
            "timestamp": pd.date_range(
                "2026-01-01", periods=2, freq="s"
            ),
            "failure_event": [0, 2],
        }
    )

    with pytest.raises(ValueError, match="only 0 or 1"):
        add_rul_target(data)


def test_rul_is_missing_when_scenario_has_no_failure():
    data = pd.DataFrame(
        {
            "scenario_source": ["scenario_1"] * 3,
            "timestamp": pd.date_range(
                "2026-01-01", periods=3, freq="s"
            ),
            "failure_event": [0, 0, 0],
        }
    )

    result = add_rul_target(data)

    assert result["rul_seconds"].isna().all()


def test_degradation_generator_produces_countdown_to_failure():
    data = generate_degradation_data(
        num_samples=10,
        sampling_interval_seconds=1,
        seed=42,
    )

    result = add_rul_target(data)

    assert len(result) == 10
    assert result["failure_event"].sum() == 1
    assert result["failure_event"].iloc[-1] == 1
    assert result["rul_seconds"].tolist() == list(
        range(9, -1, -1)
    )
