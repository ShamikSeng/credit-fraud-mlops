"""Single source of truth for the UPI dataset's schema, timeline and model settings.

Kept free of heavy imports (no pandas / mlflow / xgboost) so config.py, schemas.py and the
tests can read it cheaply.
"""

DATA_FILE = "upi_transactions.csv"
TARGET = "is_fraud"

# Days since the start of the simulated stream (float, e.g. 93.75 = day 93, 18:00). It is
# metadata, not a model input: it drives the time-based train/holdout split, and like ULB's
# Time it would show permanent drift if it were monitored.
TIME_COL = "txn_time"

# Allowed values per categorical feature. The request schema rejects anything else (422),
# and the one-hot encoder uses exactly these columns, so train and serve can't disagree.
CATEGORIES = {
    "txn_type": ["P2P", "P2M"],
    "merchant_category": ["p2p_transfer", "grocery", "food_delivery", "ecommerce", "fuel",
                          "utilities", "telecom_recharge", "travel", "entertainment",
                          "healthcare", "education", "gaming"],
    "upi_app": ["PhonePe", "GooglePay", "Paytm", "BHIM", "BankApp"],
    "device_type": ["Android", "iOS"],
    "payer_bank": ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "BoB", "Canara",
                   "Union", "Yes"],
    "state": ["Maharashtra", "Karnataka", "Uttar Pradesh", "Tamil Nadu", "Delhi", "Gujarat",
              "Telangana", "West Bengal", "Rajasthan", "Kerala", "Other"],
}

NUMERIC_FEATURES = [
    "amount",               # INR
    "hour",                 # 0-23, local time
    "day_of_week",          # 0 = Monday
    "is_new_device",        # 1 if the payer's device was never seen on this account
    "is_new_payee",         # 1 if the payer has never paid this VPA before
    "payee_vpa_age_days",   # how long the receiving VPA has existed
    "txns_last_24h",        # payer's transactions in the previous 24 h (velocity)
    "amount_to_user_avg",   # amount / payer's usual amount
]

# Model input order: what a fraud analyst reads first comes first.
FEATURES = ["amount", "hour", "day_of_week", "txn_type", "merchant_category", "upi_app",
            "device_type", "payer_bank", "state", "is_new_device", "is_new_payee",
            "payee_vpa_age_days", "txns_last_24h", "amount_to_user_avg"]

# Every model input is monitored for drift (TIME_COL is metadata and excluded).
DRIFT_FEATURES = list(FEATURES)

# --- Simulation timeline ------------------------------------------------------------------
SIM_DAYS = 180
SIM_START_DATE = "2025-01-01"
DRIFT_START_DAY = 90          # festive-sale season begins and fraudsters switch tactics
TRAIN_END_DAY = 60            # the model is trained on days [0, 60)...
HOLDOUT_END_DAY = 90          # ...and evaluated on the following, unseen days [60, 90)

EXPERIMENT_NAME = "upi_fraud_detection"

# Fixed hyperparameters. On ULB a 20-trial Hyperopt search gave no significant gain over
# defaults, so the UPI model uses one fixed configuration instead of repeating the search.
XGB_PARAMS = {
    "max_depth": 5,
    "n_estimators": 300,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "eval_metric": "aucpr",
    "n_jobs": -1,
}
