# AeroVital-IVHM — Held-Out Test Evaluation

## Evaluation protocol
- Dataset: `data/processed/test.csv`
- Test samples: 4,000
- Models were loaded from saved artifacts.
- No fitting or threshold tuning was performed by the evaluation script.
- Results apply to the current synthetic dataset.

## Fault diagnosis

Model: Random Forest Classifier

| Metric | Result |
|---|---:|
| Accuracy | 1.0000 |
| Macro F1 | 1.0000 |

| Class | Test samples | Correct predictions |
|---|---:|---:|
| Battery degradation | 50 | 50 |
| Motor imbalance | 75 | 75 |
| Motor overheating | 150 | 150 |
| Normal | 3,725 | 3,725 |

The classifier correctly classified all test samples. This result must be interpreted cautiously because the simulated fault patterns have distinct sensor signatures.

## Anomaly detection

Model: Isolation Forest using its default prediction threshold.

| Metric | Result |
|---|---:|
| Fault precision | 0.2544 |
| Fault recall | 0.9964 |
| Fault F1 | 0.4053 |
| False alarms | 803 |
| Missed faults | 1 |

Confusion matrix, rows = actual labels and columns = predicted labels:

| | Predicted normal | Predicted fault |
|---|---:|---:|
| Actual normal | 2,922 | 803 |
| Actual fault | 1 | 274 |

## Limitations and interpretation
- These are synthetic-data results, not measured real-world UAV performance.
- Perfect diagnosis accuracy may reflect the simplicity of the injected fault signatures.
- Anomaly detection generates many false alarms despite its high fault recall.
- Further robustness testing with noisier, overlapping, and more realistic fault patterns is needed.
- No remaining-useful-life (RUL) model has been validated yet.

## Next steps
1. Improve anomaly detection and test robustness under more realistic operating conditions.
2. Build an integrated prediction interface for anomaly detection and fault diagnosis.
3. Extend the simulator with degradation trajectories and defensible RUL ground truth.
4. Integrate health assessment, risk scoring, and maintenance recommendations.
