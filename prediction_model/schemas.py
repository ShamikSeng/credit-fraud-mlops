"""Request/response schemas for the serving layer.

Kept in its own module (not inline in main.py) so the schema/type validation gate
in tests can import it without pulling in FastAPI / mlflow / boto3.
"""
from pydantic import BaseModel, Field, create_model


# A credit-card transaction: Time, the 28 PCA components V1..V28, and Amount.
# All fields are strictly typed floats — this typed boundary is the input-hardening
# layer (a malformed value is rejected with HTTP 422 before it ever reaches the model),
# replacing the loan pipeline's untyped `str` categoricals + silent-NaN encoder.
_FIELD_DEFINITIONS = {name: (float, ...) for name in
                      (['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])}

TransactionFeatures: type[BaseModel] = create_model(
    'TransactionFeatures', **_FIELD_DEFINITIONS
)


# --- Response schema (Phase 3a: local SHAP explanation) ---
# Field descriptions carry the units on purpose: "what units are these SHAP values in?"
# is a direct question, and the API response must answer it without external docs.

class FeatureContribution(BaseModel):
    feature: str = Field(description="Feature name (Time, V1..V28, or Amount).")
    value: float = Field(description="The raw submitted value of this feature.")
    shap_value: float = Field(
        description="SHAP contribution in LOG-ODDS (margin) space, not probability; "
                    ">0 pushes the decision toward fraud, <0 toward legit.")
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
    status: str = Field(description="'Fraud' or 'Legitimate'.")
    fraud_probability: float = Field(
        description="P(fraud) in [0, 1] from predict_proba — the human-readable score "
                    "(distinct from the log-odds SHAP values below).")
    explanation: PredictionExplanation | None = Field(
        default=None,
        description="Feature attribution; present only for fraud-flagged transactions, "
                    "null otherwise.")
