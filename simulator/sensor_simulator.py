from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


SUPPORTED_FAULTS = {
    "motor_overheating",
    "battery_degradation",
    "motor_imbalance",
}


def generate_sensor_data(
    num_samples: int = 1000,
    sampling_interval_seconds: int = 1,
    seed: int = 42,
    fault_type: str | None = None,
    fault_start: int = 300,
    fault_duration: int = 100,
) -> pd.DataFrame:
    """Generate multivariate UAV sensor data with optional fault injection.

    Args:
        num_samples: Number of sensor readings to generate.
        sampling_interval_seconds: Time between consecutive readings.
        seed: Random seed for reproducible sensor values.
        fault_type: Optional fault to inject. Supported values are
            motor_overheating, battery_degradation, and motor_imbalance.
        fault_start: Index of the first sample affected by the fault.
        fault_duration: Number of samples affected by the fault.

    Returns:
        A DataFrame containing timestamps, sensor readings, and fault labels.
    """

    # Validate inputs.
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")

    if sampling_interval_seconds < 1:
        raise ValueError(
            "sampling_interval_seconds must be at least 1."
        )

    if fault_type is not None and fault_type not in SUPPORTED_FAULTS:
        raise ValueError(
            f"Unsupported fault_type: {fault_type}. "
            f"Choose one of {sorted(SUPPORTED_FAULTS)} or None."
        )

    if fault_start < 0:
        raise ValueError("fault_start must be non-negative.")

    if fault_duration < 1:
        raise ValueError("fault_duration must be at least 1.")

    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Generate timestamps at fixed intervals.
    # Sensor values are reproducible for the same seed and inputs.
    start_time = datetime.now(timezone.utc)
    elapsed = np.arange(num_samples) * sampling_interval_seconds

    timestamps = [
        start_time + timedelta(seconds=int(seconds))
        for seconds in elapsed
    ]

    # Shared flight-load signal creates correlations between sensors.
    flight_load = (
        0.55
        + 0.12 * np.sin(elapsed / 35)
        + rng.normal(0, 0.025, num_samples)
    )
    flight_load = np.clip(flight_load, 0.2, 0.9)

    # Environmental measurements.
    ambient_temp = (
        25
        + 1.5 * np.sin(elapsed / 180)
        + rng.normal(0, 0.25, num_samples)
    )

    pressure = (
        1013.25
        - 0.12 * (30 + 8 * np.sin(elapsed / 50))
        + rng.normal(0, 0.15, num_samples)
    )

    humidity = np.clip(
        55
        + 5 * np.sin(elapsed / 100)
        + rng.normal(0, 1.0, num_samples),
        0,
        100,
    )

    # Motor subsystem.
    motor_temp = (
        ambient_temp
        + 20 * flight_load
        + rng.normal(0, 0.8, num_samples)
    )

    vibration = (
        0.15
        + 0.65 * flight_load
        + rng.normal(0, 0.035, num_samples)
    )

    motor_current = (
        2.0
        + 18 * flight_load
        + rng.normal(0, 0.4, num_samples)
    )

    motor_rpm = (
        2500
        + 4500 * flight_load
        + rng.normal(0, 60, num_samples)
    )

    # Battery subsystem.
    battery_voltage = (
        16.8
        - 0.7 * (elapsed / max(elapsed[-1], 1))
        - 0.8 * flight_load
        + rng.normal(0, 0.08, num_samples)
    )

    battery_current = (
        motor_current
        + rng.normal(0, 0.3, num_samples)
    )

    battery_temp = (
        ambient_temp
        + 4
        + 0.15 * battery_current
        + rng.normal(0, 0.5, num_samples)
    )

    battery_soc = np.clip(
        100
        - 0.035 * elapsed
        - 2 * flight_load
        + rng.normal(0, 0.2, num_samples),
        0,
        100,
    )

    # Flight and motion measurements.
    altitude = (
        30
        + 8 * np.sin(elapsed / 50)
        + rng.normal(0, 0.6, num_samples)
    )

    speed = (
        8
        + 5 * flight_load
        + rng.normal(0, 0.3, num_samples)
    )

    acceleration = (
        0.2 * np.sin(elapsed / 12)
        + rng.normal(0, 0.05, num_samples)
    )

    # Define the fault window. The end index is exclusive.
    fault_end = min(fault_start + fault_duration, num_samples)
    fault_active = np.zeros(num_samples, dtype=bool)

    if fault_start < num_samples:
        fault_active[fault_start:fault_end] = True

    # Ground-truth labels: 0 = normal, 1 = injected fault.
    fault_label = np.zeros(num_samples, dtype=int)
    if fault_type is not None:
        fault_label[fault_active] = 1

    # Inject fault-specific changes into correlated sensor readings.
    if fault_type == "motor_overheating":
        motor_temp[fault_active] += 18
        motor_current[fault_active] += 3
        vibration[fault_active] += 0.25

    elif fault_type == "battery_degradation":
        battery_voltage[fault_active] -= 2.0
        battery_temp[fault_active] += 8
        battery_soc[fault_active] = np.clip(
            battery_soc[fault_active] - 12,
            0,
            100,
        )

    elif fault_type == "motor_imbalance":
        vibration[fault_active] += 0.8
        motor_current[fault_active] += 2.5
        motor_rpm[fault_active] -= 500

    # Calculate electrical power from voltage and current.
    electrical_power = battery_voltage * battery_current

    # Assign a readable fault label to each row.
    fault_type_labels = np.full(
        num_samples,
        "normal",
        dtype=object,
    )

    if fault_type is not None:
        fault_type_labels[fault_active] = fault_type

    # Return the generated dataset.
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "flight_load": flight_load,
            "ambient_temp_c": ambient_temp,
            "motor_temp_c": motor_temp,
            "motor_vibration": vibration,
            "motor_current_a": motor_current,
            "motor_rpm": motor_rpm,
            "battery_voltage_v": battery_voltage,
            "battery_current_a": battery_current,
            "battery_temp_c": battery_temp,
            "battery_soc_pct": battery_soc,
            "electrical_power_w": electrical_power,
            "altitude_m": altitude,
            "speed_mps": speed,
            "acceleration_mps2": acceleration,
            "pressure_hpa": pressure,
            "humidity_pct": humidity,
            "fault_label": fault_label,
            "fault_type": fault_type_labels,
        }
    )
