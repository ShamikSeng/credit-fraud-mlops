from sklearn.pipeline import Pipeline
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp


def build_pipeline(classifier=None):
    """Single source of truth for the fraud preprocessing steps.

    training_pipeline.py builds its trainable pipeline by calling this with a
    classifier appended, so the preprocessing is defined exactly once (the
    baseline's pipeline.py / training_pipeline.py duplication is removed).

    Returns fresh transformer instances on every call, so it is safe to rebuild
    once per Hyperopt trial.
    """
    steps = [
        ('MeanImputation', pp.MeanImputer(variables=config.FEATURES)),
        ('RobustScaling', pp.RobustScalerTransformer(variables=config.ROBUST_SCALE_FEATURES)),
    ]
    if classifier is not None:
        steps.append(('classifier', classifier))
    return Pipeline(steps)


# Preprocessing-only pipeline for any inference-time reuse.
preprocessing_pipeline = build_pipeline()
