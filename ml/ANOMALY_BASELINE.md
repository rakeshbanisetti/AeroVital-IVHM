# AeroVital-IVHM — Anomaly Detection Baseline

## Model
- Algorithm: Isolation Forest
- Estimators: 200
- Random seed: 42
- Training data: Normal-labeled rows from `data/processed/train.csv`
- Features: 14 numerical sensor features
- Evaluation split: `data/processed/validation.csv`
- Data type: Synthetic UAV sensor readings

## Default prediction results

| Metric | Result |
|---|---:|
| Accuracy | 80.45% |
| Fault precision | 30.71% |
| Fault recall | 98.29% |
| Fault F1 score | 0.4680 |
| False alarms | 776 |
| Missed faults | 6 |

## Threshold investigation

Five candidate thresholds were compared using validation anomaly scores.

| Threshold | Precision | Recall | F1 | False alarms | Missed faults |
|---:|---:|---:|---:|---:|---:|
| 0.027308 | 0.517 | 0.591 | 0.552 | 193 | 143 |
| 0.032416 | 0.566 | 0.517 | 0.540 | 139 | 169 |
| 0.039221 | 0.617 | 0.423 | 0.502 | 92 | 202 |
| 0.045875 | 0.688 | 0.314 | 0.431 | 50 | 240 |
| 0.058893 | 0.750 | 0.171 | 0.279 | 20 | 290 |

The 0.027308 threshold had the highest F1 score among the five candidates. It remains a candidate, not a finalized operational threshold.

## Limitations
- Results are from synthetic data and do not establish real-world UAV performance.
- Synthetic fault patterns may be easier to detect than real faults.
- Lowering false alarms can increase missed faults.
- The threshold was explored on validation data; final test evaluation remains pending.
- The current model detects unusual readings but does not identify the fault type.

## Next steps
1. Establish a threshold-selection criterion appropriate for the intended monitoring use.
2. Finalize the anomaly detector without using the test split for tuning.
3. Evaluate once on the held-out test split.
4. Build a separate supervised model for fault diagnosis.
