"""Train the UPI fraud model, log it to MLflow, and write its drift reference.

Run from the project root (after `python -m prediction_model.upi.simulate`):
    python -m prediction_model.upi.train

The model is trained on days [0, TRAIN_END_DAY) and scored once on the following, unseen days
[TRAIN_END_DAY, HOLDOUT_END_DAY). The split is by TIME, not random: in a payment stream a
model is always used on transactions that happen after its training data, and a random split
would let it peek at the future.

It is logged to its own MLflow experiment with the same metric names as ULB (pr_auc, ...), so
the API serves it exactly the same way: the run with the best holdout PR-AUC.
"""
import os

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

import mlflow
import mlflow.sklearn
from prediction_model.config import config
from prediction_model.fix_mlruns_paths import fix_mlruns_paths
from prediction_model.upi import modeling, spec


def load_stream():
    return pd.read_csv(os.path.join(config.DATAPATH, spec.DATA_FILE))


def main():
    df = load_stream()
    train = modeling.time_slice(df, 0, spec.TRAIN_END_DAY)
    holdout = modeling.time_slice(df, spec.TRAIN_END_DAY, spec.HOLDOUT_END_DAY)
    model = modeling.fit_model(train)

    scores = modeling.evaluate(model, holdout)
    proba = model.predict_proba(holdout[spec.FEATURES])[:, 1]
    y = holdout[spec.TARGET]
    metrics = {
        "pr_auc": scores["pr_auc"],
        "roc_auc": roc_auc_score(y, proba),
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": f1_score(y, proba >= 0.5),
    }

    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(spec.EXPERIMENT_NAME)
    with mlflow.start_run(run_name="upi_xgb_time_split"):
        mlflow.log_params({**spec.XGB_PARAMS, "train_days": f"0-{spec.TRAIN_END_DAY}",
                           "holdout_days": f"{spec.TRAIN_END_DAY}-{spec.HOLDOUT_END_DAY}"})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, config.MODEL_NAME.lstrip("/"))

    # Drift reference = the raw inputs the model was trained on (seeded sample), as for ULB.
    reference = train[spec.DRIFT_FEATURES]
    if len(reference) > config.DRIFT_REFERENCE_SAMPLE:
        reference = reference.sample(n=config.DRIFT_REFERENCE_SAMPLE,
                                     random_state=config.RANDOM_STATE)
    reference.to_csv(config.UPI_DRIFT_REFERENCE_FILE, index=False)
    fix_mlruns_paths()   # keep mlruns portable between the host and the containers

    print(f"Trained on days [0, {spec.TRAIN_END_DAY}): {len(train):,} rows, "
          f"{int(train[spec.TARGET].sum())} frauds")
    print(f"Holdout days [{spec.TRAIN_END_DAY}, {spec.HOLDOUT_END_DAY}): {len(holdout):,} rows, "
          f"{scores['n_fraud']} frauds")
    print("  " + ", ".join(f"{k} {v:.3f}" for k, v in metrics.items())
          + f"  ({scores['tp']} caught, {scores['n_fraud'] - scores['tp']} missed, "
            f"{scores['fp']} false alarms)")
    print(f"Wrote drift reference: {config.UPI_DRIFT_REFERENCE_FILE} ({len(reference):,} rows)")


if __name__ == "__main__":
    main()
