"""
Simple sanity tests for pipeline outputs and files.
"""

import os
import json


def test_dataset_exists():
    path = os.path.join("data", "raw", "online_shoppers_intention.csv")
    assert os.path.exists(path), "Raw dataset file must exist."


def test_model_saved():
    model_path = os.path.join("models", "spark_purchase_model")
    assert os.path.exists(model_path), "Trained MLlib model directory must exist."


def test_metrics_json():
    metrics_path = os.path.join("data", "processed", "model_metrics.json")
    assert os.path.exists(metrics_path), "Model metrics JSON must exist."
    with open(metrics_path, "r") as f:
        m = json.load(f)
    assert "roc_auc" in m and m["roc_auc"] > 0.85, "Model ROC-AUC should exceed 0.85"


def test_batch_summary_csv():
    summary_path = os.path.join("data", "processed", "batch_monthly_summary.csv")
    assert os.path.exists(summary_path), "Monthly summary CSV must exist."
