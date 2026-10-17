import pandas as pd


def add_rul_target(data: pd.DataFrame) -> pd.DataFrame:
    """Add remaining seconds until the failure event for each scenario."""
    required_columns = {
        "scenario_source",
        "timestamp",
        "failure_event",
    }
    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if data.empty:
        raise ValueError("Input data must not be empty.")

    result = data.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result["failure_event"] = result["failure_event"].astype(int)

    if not result["failure_event"].isin([0, 1]).all():
        raise ValueError("failure_event must contain only 0 or 1.")

    result["rul_seconds"] = pd.NA

    for _, indices in result.groupby("scenario_source", sort=False).groups.items():
        scenario = result.loc[indices].sort_values("timestamp")
        failures = scenario.index[scenario["failure_event"] == 1]

        if len(failures) == 0:
            continue

        failure_index = failures[0]
        failure_time = scenario.loc[failure_index, "timestamp"]

        before_failure = scenario["timestamp"] <= failure_time
        rul = (
            failure_time - scenario.loc[before_failure, "timestamp"]
        ).dt.total_seconds()

        result.loc[rul.index, "rul_seconds"] = rul

    result["rul_seconds"] = pd.to_numeric(result["rul_seconds"])
    return result
