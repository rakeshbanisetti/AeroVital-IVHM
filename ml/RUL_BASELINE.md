# RUL Regression Baseline

## Dataset and split
- Dataset: synthetic motor degradation scenarios
- Total rows: 30000
- Total scenarios: 30
- Training rows: 24000
- Test rows: 6000
- Held-out test scenarios: 6
- Features: motor_temp_c, motor_vibration, motor_current_a, motor_rpm
- Target: `rul_seconds` in seconds

## Held-out scenario metrics
- Mean Absolute Error (MAE): 71.683 seconds
- Root Mean Squared Error (RMSE): 86.406 seconds
- R-squared: 0.9104

## Limitations
This is an initial baseline evaluated on synthetic scenarios. The
degradation patterns and failure times are simulated, not validated
against real UAV operating or maintenance data. These metrics do not
establish real-world remaining-useful-life accuracy.
