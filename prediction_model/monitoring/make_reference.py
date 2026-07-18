"""Regenerate the drift reference = the model's TRAINING input distribution.

Run standalone:  python -m prediction_model.monitoring.make_reference

Decoupled from training_pipeline.py on purpose: the stratified split is deterministic
(seed = config.RANDOM_STATE), so re-running the same split here reproduces the exact X_train
the final model was fit on — without importing MLflow/Hyperopt or re-training. The reference
is written to config.DRIFT_REFERENCE_FILE as a regenerable build artifact (git-ignored, like
the trained model), so it is never a stale committed snapshot.

Feature space: raw config.DRIFT_FEATURES (V1..V28 + Amount) — pre-scaling, since we monitor
the raw inputs the API receives, and excluding the monotonic `Time` column and the `Class`
label (see config for the rationale).
"""
import os

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, stratified_split


def build_reference():
    df = load_dataset(config.DATA_FILE)
    # Same deterministic split the model trained on; we only need X_train.
    X_train, _X_test, _y_train, _y_test = stratified_split(df)

    reference = X_train[config.DRIFT_FEATURES]

    # Seeded sample to keep the artifact small and drift compute fast; if the training split
    # is already smaller than the cap, keep it whole.
    if len(reference) > config.DRIFT_REFERENCE_SAMPLE:
        reference = reference.sample(n=config.DRIFT_REFERENCE_SAMPLE,
                                     random_state=config.RANDOM_STATE)
    return reference


def main():
    reference = build_reference()
    os.makedirs(config.MONITORING_DIR, exist_ok=True)
    reference.to_csv(config.DRIFT_REFERENCE_FILE, index=False)
    print(f"Wrote drift reference: {config.DRIFT_REFERENCE_FILE}")
    print(f"  rows={len(reference)}  cols={len(reference.columns)}  "
          f"(features={list(reference.columns[:3])}...{list(reference.columns[-1:])})")


if __name__ == "__main__":
    main()
