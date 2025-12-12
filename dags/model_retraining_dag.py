from __future__ import annotations

"""
Hospital Capacity Model Retraining DAG

What this DAG does
- Retrains on a monthly schedule
- Evaluates candidate vs current production
- Only promotes the candidate if promotion criteria pass

Notes
- Uses SQLite for demo purposes
- Stores artifacts locally in MODEL_DIR (Airflow workers need shared storage)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

from datetime import timedelta
import json
import os
import shutil
import sqlite3

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss


# Configuration
MODEL_DIR = os.getenv('MODEL_DIR', '/opt/airflow/models')
DATA_DB = os.getenv('DATA_DB', '/opt/airflow/data/hospital_capacity.db')

PROD_MODEL_PATH = os.path.join(MODEL_DIR, 'production_model.pkl')
CANDIDATE_MODEL_PATH = os.path.join(MODEL_DIR, 'candidate_model.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')


# Auto-promotion thresholds
AUC_IMPROVEMENT_THRESHOLD = float(os.getenv('AUC_IMPROVEMENT_THRESHOLD', '0.01'))
RECALL_REGRESSION_THRESHOLD = float(os.getenv('RECALL_REGRESSION_THRESHOLD', '0.10'))
DRIFT_KS_THRESHOLD = float(os.getenv('DRIFT_KS_THRESHOLD', '0.01'))
DRIFT_PSI_THRESHOLD = float(os.getenv('DRIFT_PSI_THRESHOLD', '0.25'))


def _safe_json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f_handle:
        json.dump(obj, f_handle, indent=2, default=str)


def extract_training_data(**context):
    conn = sqlite3.connect(DATA_DB)
    query = "SELECT * FROM hospital_capacity_features WHERE date >= date('now', '-12 months') ORDER BY date ASC"
    df_train = pd.read_sql_query(query, conn)
    conn.close()

    if len(df_train) == 0:
        raise ValueError('No training data returned from hospital_capacity_features')

    context['ti'].xcom_push(key='training_data', value=df_train.to_json(orient='records'))
    return int(len(df_train))


def train_candidate_model(**context):
    ti = context['ti']
    df_json = ti.xcom_pull(key='training_data', task_ids='extract_training_data')
    df_train = pd.read_json(df_json, orient='records')

    feature_cols = [col for col in df_train.columns if col not in ['date', 'hospital_id', 'high_capacity_flag']]
    if 'high_capacity_flag' not in df_train.columns:
        raise ValueError('Expected label column high_capacity_flag not found')

    X = df_train[feature_cols]
    y = df_train['high_capacity_flag']

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, CANDIDATE_MODEL_PATH)

    ti.xcom_push(key='feature_cols', value=json.dumps(feature_cols))
    return int(len(X))


def _psi(ref, cur, bins=10):
    ref = pd.Series(ref).replace([np.inf, -np.inf], np.nan).dropna()
    cur = pd.Series(cur).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ref) < 10 or len(cur) < 10:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(ref.quantile(quantiles).values)
    if len(cuts) < 3:
        return 0.0

    ref_bins = pd.cut(ref, bins=cuts, include_lowest=True)
    cur_bins = pd.cut(cur, bins=cuts, include_lowest=True)

    ref_dist = ref_bins.value_counts(normalize=True).sort_index()
    cur_dist = cur_bins.value_counts(normalize=True).sort_index()

    cur_dist = cur_dist.reindex(ref_dist.index).fillna(0.0)

    eps = 1e-6
    ref_p = np.clip(ref_dist.values, eps, None)
    cur_p = np.clip(cur_dist.values, eps, None)

    psi_val = float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
    return psi_val


def calculate_drift(**context):
    """
    Drift is computed between recent window and older window from the same extracted training data.
    This keeps the demo self-contained without needing a separate production reference dataset.
    """
    ti = context['ti']
    df_json = ti.xcom_pull(key='training_data', task_ids='extract_training_data')
    df_train = pd.read_json(df_json, orient='records')

    feature_cols = json.loads(ti.xcom_pull(key='feature_cols', task_ids='train_candidate_model'))

    df_sorted = df_train.sort_values('date')
    split_idx = int(len(df_sorted) * 0.7)
    df_ref = df_sorted.iloc[:split_idx]
    df_cur = df_sorted.iloc[split_idx:]

    ks_pvals = {}
    psi_scores = {}

    for col in feature_cols:
        ref_vals = df_ref[col].replace([np.inf, -np.inf], np.nan).dropna()
        cur_vals = df_cur[col].replace([np.inf, -np.inf], np.nan).dropna()

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            ks_pvals[col] = 1.0
            psi_scores[col] = 0.0
            continue

        ks_p = float(stats.ks_2samp(ref_vals, cur_vals).pvalue)
        ks_pvals[col] = ks_p
        psi_scores[col] = _psi(ref_vals, cur_vals)

    drift_summary = {
        'ks_pvals': ks_pvals,
        'psi_scores': psi_scores,
        'max_psi': float(max(psi_scores.values())) if len(psi_scores) > 0 else 0.0,
        'min_ks_pvalue': float(min(ks_pvals.values())) if len(ks_pvals) > 0 else 1.0
    }

    ti.xcom_push(key='drift_summary', value=json.dumps(drift_summary))
    return drift_summary


def evaluate_models(**context):
    ti = context['ti']
    df_json = ti.xcom_pull(key='training_data', task_ids='extract_training_data')
    df_all = pd.read_json(df_json, orient='records')

    feature_cols = json.loads(ti.xcom_pull(key='feature_cols', task_ids='train_candidate_model'))

    df_all = df_all.sort_values('date')
    split_idx = int(len(df_all) * 0.8)
    df_eval = df_all.iloc[split_idx:]

    X_eval = df_eval[feature_cols]
    y_eval = df_eval['high_capacity_flag']

    candidate_model = joblib.load(CANDIDATE_MODEL_PATH)

    prod_exists = os.path.exists(PROD_MODEL_PATH)
    production_model = joblib.load(PROD_MODEL_PATH) if prod_exists else None

    cand_proba = candidate_model.predict_proba(X_eval)[:, 1]
    cand_pred = (cand_proba >= 0.5).astype(int)

    cand_metrics = {
        'auc': float(roc_auc_score(y_eval, cand_proba)) if len(np.unique(y_eval)) > 1 else 0.0,
        'precision': float(precision_score(y_eval, cand_pred, zero_division=0)),
        'recall': float(recall_score(y_eval, cand_pred, zero_division=0)),
        'brier': float(brier_score_loss(y_eval, cand_proba))
    }

    if production_model is None:
        prod_metrics = {
            'auc': None,
            'precision': None,
            'recall': None,
            'brier': None
        }
    else:
        prod_proba = production_model.predict_proba(X_eval)[:, 1]
        prod_pred = (prod_proba >= 0.5).astype(int)
        prod_metrics = {
            'auc': float(roc_auc_score(y_eval, prod_proba)) if len(np.unique(y_eval)) > 1 else 0.0,
            'precision': float(precision_score(y_eval, prod_pred, zero_division=0)),
            'recall': float(recall_score(y_eval, prod_pred, zero_division=0)),
            'brier': float(brier_score_loss(y_eval, prod_proba))
        }

    drift_summary = json.loads(ti.xcom_pull(key='drift_summary', task_ids='calculate_drift'))

    metrics_payload = {
        'candidate': cand_metrics,
        'production': prod_metrics,
        'drift': drift_summary
    }

    _safe_json_dump(metrics_payload, METRICS_PATH)
    ti.xcom_push(key='metrics', value=json.dumps(metrics_payload))
    return metrics_payload


def decide_promotion(**context):
    """Return downstream task_id for Airflow branching."""
    ti = context['ti']
    metrics = json.loads(ti.xcom_pull(key='metrics', task_ids='evaluate_models'))

    cand = metrics['candidate']
    prod = metrics['production']
    drift = metrics['drift']

    drift_bad = (drift.get('min_ks_pvalue', 1.0) < DRIFT_KS_THRESHOLD) or (drift.get('max_psi', 0.0) > DRIFT_PSI_THRESHOLD)

    # If there is no production model yet, promote the first candidate as a bootstrap.
    if prod.get('auc') is None:
        return 'promote_model'

    auc_improvement = float(cand['auc'] - float(prod['auc']))
    recall_regression = float(float(prod['recall']) - float(cand['recall']))

    pass_auc = auc_improvement >= AUC_IMPROVEMENT_THRESHOLD
    pass_recall = recall_regression <= RECALL_REGRESSION_THRESHOLD

    if (not drift_bad) and pass_auc and pass_recall:
        return 'promote_model'
    return 'skip_promotion'


def promote_model(**context):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(CANDIDATE_MODEL_PATH):
        raise FileNotFoundError('Candidate model not found at ' + CANDIDATE_MODEL_PATH)

    # Backup current production model
    if os.path.exists(PROD_MODEL_PATH):
        backup_path = os.path.join(MODEL_DIR, 'production_model.backup.pkl')
        shutil.copy2(PROD_MODEL_PATH, backup_path)

    shutil.copy2(CANDIDATE_MODEL_PATH, PROD_MODEL_PATH)
    return True


default_args = {
    'owner': 'ml-platform',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}


with DAG(
    dag_id='hospital_capacity_model_retraining',
    default_args=default_args,
    description='Monthly retraining with drift detection, evaluation, and gated auto-promotion',
    start_date=days_ago(1),
    schedule_interval='0 2 1 * *',
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'retraining', 'hospital-capacity']
) as dag:

    extract_task = PythonOperator(
        task_id='extract_training_data',
        python_callable=extract_training_data
    )

    train_task = PythonOperator(
        task_id='train_candidate_model',
        python_callable=train_candidate_model
    )

    drift_task = PythonOperator(
        task_id='calculate_drift',
        python_callable=calculate_drift
    )

    eval_task = PythonOperator(
        task_id='evaluate_models',
        python_callable=evaluate_models
    )

    decide_task = BranchPythonOperator(
        task_id='decide_promotion',
        python_callable=decide_promotion
    )

    promote_task = PythonOperator(
        task_id='promote_model',
        python_callable=promote_model
    )

    skip_task = EmptyOperator(task_id='skip_promotion')

    done_task = EmptyOperator(task_id='done')

    extract_task >> train_task >> drift_task >> eval_task >> decide_task
    decide_task >> promote_task >> done_task
    decide_task >> skip_task >> done_task
