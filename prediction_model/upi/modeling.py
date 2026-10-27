"""Model building, evaluation and time slicing for the UPI stream (no MLflow, no Evidently),
shared by the training script, the demo-asset script and the tests.
"""
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from prediction_model.processing import preprocessing as pp
from prediction_model.upi import spec


def build_pipeline(classifier=None):
    """Mean imputation on numeric inputs, then fixed-category one-hot encoding. Trees need
    no scaling, so unlike ULB there is no scaler step."""
    steps = [
        ("MeanImputation", pp.MeanImputer(variables=spec.NUMERIC_FEATURES)),
        ("OneHot", pp.OneHotTransformer(categories=spec.CATEGORIES)),
    ]
    if classifier is not None:
        steps.append(("classifier", classifier))
    return Pipeline(steps)


def fit_model(df, seed=42):
    """Fit a fresh pipeline on labelled rows, with the positive-class weight set to
    n_legit / n_fraud of these rows (cost-sensitive, as on ULB)."""
    X, y = df[spec.FEATURES], df[spec.TARGET]
    weight = (y == 0).sum() / max((y == 1).sum(), 1)
    clf = xgb.XGBClassifier(**spec.XGB_PARAMS, scale_pos_weight=weight, random_state=seed)
    model = build_pipeline(clf)
    model.fit(X, y)
    return model


def evaluate(model, df, threshold=0.5):
    """PR-AUC plus fraud-class precision/recall at the threshold. PR-AUC is NaN when the
    slice has no frauds (it is undefined, not zero)."""
    y = df[spec.TARGET].to_numpy()
    n_fraud = int(y.sum())
    proba = model.predict_proba(df[spec.FEATURES])[:, 1]
    flagged = proba >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    return {
        "n": int(len(df)),
        "n_fraud": n_fraud,
        "pr_auc": float(average_precision_score(y, proba)) if n_fraud else float("nan"),
        "recall": tp / n_fraud if n_fraud else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "tp": tp,
        "fp": fp,
    }


def time_slice(df, start, end):
    t = df[spec.TIME_COL]
    return df[(t >= start) & (t < end)]
