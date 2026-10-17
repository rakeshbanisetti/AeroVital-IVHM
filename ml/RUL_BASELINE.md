# RUL Regression Baseline

## Dataset and split
- Dataset: synthetic motor degradation scenarios
- Total rows: 30000
- Total scenarios: 30
- Training rows: 24000
- Test rows: 6000
- Held-out test scenarios: 6
- Features: motor_temp_c, motor_vibration, motor_current_a, motor_rpm, degradation_level
- Target: `rul_seconds` in seconds

## Held-out scenario metrics
- Mean Absolute Error (MAE): 56.927 seconds
- Root Mean Squared Error (RMSE): 72.743 seconds
- R-squared: 0.9365

## Limitations
This is an initial baseline evaluated on synthetic scenarios. The
degradation patterns and failure times are simulated, not validated
against real UAV operating or maintenance data. These metrics do not
establish real-world remaining-useful-life accuracy.
