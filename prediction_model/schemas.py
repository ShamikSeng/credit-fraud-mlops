"""Request/response schemas for the serving layer.

Kept in its own module (not inline in main.py) so the schema/type validation gate
in tests can import it without pulling in FastAPI / mlflow / boto3.
"""
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

from prediction_model.config import config


def build_transaction_model(features, categorical=None, metadata=None,
                            name="TransactionFeatures"):
    """Build the typed request model from the feature list, so the schema can never drift
    away from what the model is trained on.

    Numeric features are required floats. Categorical features are required and must be
    one of their allowed values. Metadata fields (e.g. the UPI stream timestamp) are
    optional floats that are accepted but never fed to the model. Anything malformed is
    rejected with HTTP 422 before it reaches the model.
    """
    categorical = categorical or {}
    fields = {}
    for feature in features:
        if feature in categorical:
            fields[feature] = (Literal[tuple(categorical[feature])], ...)
        else:
            fields[feature] = (float, ...)
    for field in metadata or []:
        fields[field] = (Optional[float], None)
    return create_model(name, **fields)


# ULB: Time, the 28 PCA components V1..V28 and Amount, all strictly typed floats.
# UPI: named numeric fields plus validated categories (see prediction_model/upi/spec.py).
TransactionFeatures: type[BaseModel] = build_transaction_model(
    config.FEATURES, config.CATEGORICAL_FEATURES, config.METADATA_FIELDS)


# --- Response schema (Phase 3a: local SHAP explanation) ---
# Field descriptions carry the units on purpose: "what units are these SHAP values in?"
# is a direct question, and the API response must answer it without external docs.

class FeatureContribution(BaseModel):
    feature: str = Field(description="Input feature name, e.g. V14 (ULB) or upi_app (UPI).")
    value: Union[float, str] = Field(
        description="The raw submitted value of this feature (a category name for "
                    "categorical features).")
    shap_value: float = Field(
        description="SHAP contribution in LOG-ODDS (margin) space, not probability; "
                    ">0 pushes the decision toward fraud, <0 toward legit. For a categorical "
                    "feature this is the sum over its one-hot columns.")
    direction: str = Field(
        description="'increases_fraud_risk' or 'decreases_fraud_risk' (sign of shap_value).")


class PredictionExplanation(BaseModel):
    base_value: float = Field(
        description="Explainer expected value = the model's BASE log-odds margin before "
                    "any feature contributions. base_value + sum(ALL shap values) = model margin.")
    top_features: list[FeatureContribution] = Field(
        description="Top-N features by |shap_value|, descending — a truncation of the full "
                    "decomposition, i.e. the largest drivers, not every feature.")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())   # allow the `model_version` field name

    status: str = Field(description="'Fraud' or 'Legitimate'.")
    fraud_probability: float = Field(
        description="P(fraud) in [0, 1] from predict_proba — the human-readable score "
                    "(distinct from the log-odds SHAP values below).")
    explanation: PredictionExplanation | None = Field(
        default=None,
        description="Feature attribution; present only for fraud-flagged transactions, "
                    "null otherwise.")
    model_version: Optional[str] = Field(
        default=None,
        description="MLflow run id of the model that answered (the best run by PR-AUC).")
