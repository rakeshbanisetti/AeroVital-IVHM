## Health Assessment and Risk Classification

AeroVital-IVHM combines anomaly detection and fault diagnosis outputs with a rule-based health assessment.

### Assessment rules

| Condition | Health score | Risk level |
|---|---:|---|
| Anomaly and diagnosis disagree | 50 | Uncertain |
| Anomaly detected and fault diagnosed | 30 | High |
| Anomaly detected without a diagnosed fault | 50 | Medium |
| No anomaly detected | 95 | Low |

### Maintenance recommendations

The API provides a maintenance recommendation based on the assessment outcome. Recommendations include further sensor checks when model outputs disagree, fault-specific inspection when a fault is identified, and routine monitoring when no anomaly is detected.

**Current limitation:** These scores and rules are provisional engineering heuristics. They are not calibrated probabilities, validated physical health measurements, or evidence of real-world UAV performance. The ML models have been trained and evaluated using synthetic sensor data.
