import pathlib
import os


current_directory = os.path.dirname(os.path.realpath(__file__)) #current directory of the script

PACKAGE_ROOT = os.path.dirname(current_directory) #parent directory of current directory


# PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

# Single raw file (Kaggle/ULB Credit Card Fraud). The train/test split is done
# in-code (stratified) — see prediction_model.processing.data_handling.stratified_split —
# so there is no separate train.csv / test.csv anymore.
DATA_FILE = 'creditcard.csv'

TARGET = 'Class'  # 1 = fraud, 0 = legit

# Feature order matches the raw CSV: Time, V1..V28 (PCA components), Amount.
FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# V1..V28 are already PCA-transformed (centered/scaled); only the two raw,
# heavily-skewed columns need scaling. RobustScaler is used (median/IQR) so the
# extreme Amount outliers don't distort the scaling the way MinMax/StandardScaler would.
ROBUST_SCALE_FEATURES = ['Amount', 'Time']

# --- Split / CV / reproducibility ---
TEST_SIZE = 0.2       # stratified holdout fraction
RANDOM_STATE = 42
CV_FOLDS = 5          # StratifiedKFold folds inside the Hyperopt objective (separate from the holdout)
MAX_EVALS = 20        # Hyperopt trials

# --- Explainability (Phase 3a: local SHAP on the serving path) ---
SHAP_TOP_N = 5              # number of top contributing features returned per flagged transaction
EXPLAIN_ONLY_FLAGGED = True # explanations attached only to fraud-flagged predictions (cost scales with the ~0.17% flag rate, not total traffic)

# --- Infra (left as-is this phase; DVC/S3 flow de-prioritized, see Decisions Log) ---
S3_BUCKET = "loanprediction"

FOLDER="datadrift"

# MLflow tracking. Defaults to a local file store so training/serving work without
# the (now-unreachable) baseline EC2 server; override with MLFLOW_TRACKING_URI.
# Original baseline URI kept for reference:
#   http://ec2-3-17-191-241.us-east-2.compute.amazonaws.com:5000/
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")


EXPERIMENT_NAME="fraud_detection_model"

MODEL_NAME="/fraud-detection-model"
