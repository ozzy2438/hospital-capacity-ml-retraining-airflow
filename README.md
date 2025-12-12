# Hospital Capacity ML Retraining with Airflow

Apache Airflow pipeline for scheduled model retraining with drift detection, challenger-vs-champion evaluation, and gated auto-promotion.

## Contents

- `dags/model_retraining_dag.py` - Monthly retraining DAG with drift detection, evaluation, and gated promotion

## How promotion works

The candidate model is promoted only when all are true

- Candidate AUC improves by at least 0.01 over production
- Candidate recall does not regress by more than 0.10
- Drift checks are within thresholds (KS and PSI)

If no production model exists yet, the first candidate is promoted to bootstrap.

## Configuration

Environment variables

- `MODEL_DIR` default `/opt/airflow/models`
- `DATA_DB` default `/opt/airflow/data/hospital_capacity.db`
- `AUC_IMPROVEMENT_THRESHOLD` default `0.01`
- `RECALL_REGRESSION_THRESHOLD` default `0.10`
- `DRIFT_KS_THRESHOLD` default `0.01`
- `DRIFT_PSI_THRESHOLD` default `0.25`
