import numpy as np
import pandas as pd


def generate_degradation_data(
    num_samples: int = 1000,
    sampling_interval_seconds: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic motor degradation ending in a simulated failure."""
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2.")

    if sampling_interval_seconds < 1:
        raise ValueError("sampling_interval_seconds must be positive.")

    rng = np.random.default_rng(seed)

    elapsed = np.arange(num_samples) * sampling_interval_seconds
    progress = np.linspace(0.0, 1.0, num_samples)

    # Vary degradation speed and curve shape between scenarios.
    exponent = rng.uniform(0.7, 1.8)
    degradation = progress**exponent

    motor_temp = (
        40 + rng.uniform(20, 30) * degradation
        + rng.normal(0, 1.0, num_samples)
    )
    motor_vibration = (
        0.1 + rng.uniform(0.5, 0.9) * degradation
        + rng.normal(0, 0.025, num_samples)
    )
    motor_current = (
        5 + rng.uniform(2.0, 3.0) * degradation
        + rng.normal(0, 0.15, num_samples)
    )
    motor_rpm = (
        4500 - rng.uniform(450, 750) * degradation
        + rng.normal(0, 20, num_samples)
    )

    failure_event = np.zeros(num_samples, dtype=int)
    failure_event[-1] = 1

    return pd.DataFrame(
        {
            "timestamp": pd.Timestamp.now(tz="UTC")
            + pd.to_timedelta(elapsed, unit="s"),
            "scenario_source": f"rul_scenario_seed_{seed}",
            "motor_temp_c": motor_temp,
            "motor_vibration": motor_vibration,
            "motor_current_a": motor_current,
            "motor_rpm": motor_rpm,
            "degradation_level": degradation,
            "failure_event": failure_event,
        }
    )
