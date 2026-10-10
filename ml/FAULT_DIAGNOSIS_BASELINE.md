# AeroVital-IVHM — Fault Diagnosis Baseline

## Model configuration
- Algorithm: Random Forest Classifier
- Number of trees: 300
- Class weighting: Balanced
- Random seed: 42
- Training samples: 12,000
- Validation samples: 4,000
- Input features: 14 numerical sensor features
- Target: `fault_type`

## Validation results

| Metric | Result |
|---|---:|
| Accuracy | 100.00% |
| Macro F1 | 1.0000 |

### Per-class results

| Fault class | Validation samples | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Battery degradation | 150 | 1.00 | 1.00 | 1.00 |
| Motor imbalance | 150 | 1.00 | 1.00 | 1.00 |
| Motor overheating | 50 | 1.00 | 1.00 | 1.00 |
| Normal | 3,650 | 1.00 | 1.00 | 1.00 |

## Interpretation
The model correctly classified every validation sample in this experiment.

This is a preliminary result on a synthetic dataset. The injected faults have deliberately distinct sensor signatures, so this score may overstate performance on real UAV telemetry.

## Limitations
- Real UAV sensor data has not yet been evaluated.
- Fault signatures may be simpler than real-world failures.
- The held-out test dataset has not yet been evaluated.
- Validation results must not be presented as final test performance.

## Next steps
1. Perform final evaluation on the held-out test set after the model configuration is frozen.
2. Inspect performance by fault class and document limitations.
3. Integrate anomaly detection and diagnosis into a common prediction interface.
4. Extend the simulator with realistic degradation trajectories before training an RUL model.
