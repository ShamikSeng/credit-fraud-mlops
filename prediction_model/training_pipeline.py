import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, recall_score, precision_score)
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
import xgboost as xgb
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, stratified_split
import prediction_model.pipeline as pipe


mlflow.set_tracking_uri(config.TRACKING_URI)

# --- Data + the single stratified holdout (CV folds below are separate, and live only in X_train) ---
df = load_dataset(config.DATA_FILE)
X_train, X_test, y_train, y_test = stratified_split(df)

# Cost-sensitive weighting for the ~0.17% positive class: n_negatives / n_positives (~578).
SCALE_POS_WEIGHT = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f"scale_pos_weight = {SCALE_POS_WEIGHT:.1f} "
      f"({(y_train == 1).sum()} frauds / {(y_train == 0).sum()} legit in train)")


search_space = {
    'max_depth': hp.choice('max_depth', np.arange(3, 10, dtype=int)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'n_estimators': hp.choice('n_estimators', np.arange(100, 500, 50, dtype=int)),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'gamma': hp.uniform('gamma', 0, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    # max_delta_step helps convergence on highly imbalanced logistic problems.
    'max_delta_step': hp.quniform('max_delta_step', 0, 10, 1),
}


def make_classifier(params):
    # NOTE: use_label_encoder was removed in XGBoost >= 1.6 (pinned here at 2.0.3),
    # so it is intentionally not passed — passing it raises at construction.
    return xgb.XGBClassifier(
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        n_estimators=int(params['n_estimators']),
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        gamma=params['gamma'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        max_delta_step=int(params['max_delta_step']),
        scale_pos_weight=SCALE_POS_WEIGHT,
        eval_metric='aucpr',
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )


def objective(params):
    """Average PR-AUC across StratifiedKFold folds (not a single noisy holdout).
    Hyperopt minimises, so we return -mean(PR-AUC)."""
    clf = make_classifier(params)
    classification_pipeline = pipe.build_pipeline(clf)

    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                          random_state=config.RANDOM_STATE)
    scores = cross_val_score(classification_pipeline, X_train, y_train,
                             cv=skf, scoring='average_precision', n_jobs=1)
    mean_pr_auc = float(scores.mean())
    std_pr_auc = float(scores.std())

    with mlflow.start_run(nested=True):
        mlflow.log_params({
            'max_depth': int(params['max_depth']),
            'n_estimators': int(params['n_estimators']),
            'max_delta_step': int(params['max_delta_step']),
            'learning_rate': params['learning_rate'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'gamma': params['gamma'],
            'reg_alpha': params['reg_alpha'],
            'reg_lambda': params['reg_lambda'],
        })
        mlflow.log_metric('cv_pr_auc_mean', mean_pr_auc)
        mlflow.log_metric('cv_pr_auc_std', std_pr_auc)

    return {'loss': -mean_pr_auc, 'status': STATUS_OK}


if __name__ == '__main__':
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    with mlflow.start_run(run_name='fraud_xgb_hyperopt'):
        trials = Trials()
        best = fmin(fn=objective, space=search_space, algo=tpe.suggest,
                    max_evals=config.MAX_EVALS, trials=trials)
        best_params = space_eval(search_space, best)
        print("Best hyperparameters:", best_params)

        # Refit the best config on the full training split and evaluate ONCE on the
        # untouched holdout. Preprocessing is fit inside the pipeline on X_train only,
        # so no information from X_test leaks into scaling/imputation.
        final_pipeline = pipe.build_pipeline(make_classifier(best_params))
        final_pipeline.fit(X_train, y_train)

        proba = final_pipeline.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = {
            'pr_auc': average_precision_score(y_test, proba),   # primary / model-selection metric
            'roc_auc': roc_auc_score(y_test, proba),
            'precision': precision_score(y_test, preds, zero_division=0),
            'recall': recall_score(y_test, preds, zero_division=0),
            'f1': f1_score(y_test, preds, zero_division=0),
        }
        print("Holdout metrics:", {k: round(v, 4) for k, v in metrics.items()})

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(final_pipeline, config.MODEL_NAME.lstrip('/'))
