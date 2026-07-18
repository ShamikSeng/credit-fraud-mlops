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

# --- Drift monitoring (Phase 3b: EvidentlyAI on inference inputs -> Prometheus) ---
MONITORING_DIR = os.path.join(PACKAGE_ROOT, "monitoring")

# The drift reference is the TRAINING input distribution: raw X_train from the same
# deterministic stratified_split (seed = RANDOM_STATE) the model was fit on. It is a
# regenerated build artifact (see monitoring/make_reference.py), git-ignored like the model.
DRIFT_REFERENCE_FILE = os.path.join(MONITORING_DIR, "reference_data.csv")

# Monitor the 28 PCA components + Amount, but NOT Time: Time is seconds-since-first-
# transaction (a monotonic counter), so on any live feed it always exceeds the training
# range and would register permanent, meaningless drift. Class is excluded too — it is a
# label, not an inference input (target drift needs ground truth, a separate concern).
DRIFT_FEATURES = [f for f in FEATURES if f != 'Time']

# Window-triggered auto-recompute: buffer inference inputs and recompute drift vs the
# reference every DRIFT_WINDOW inputs (env-overridable so tests/demos can trigger fast).
DRIFT_WINDOW = int(os.getenv("DRIFT_WINDOW", "500"))

# Master switch for the auto-recompute on the serving path (the manual endpoint still works).
DRIFT_ENABLED = os.getenv("DRIFT_ENABLED", "true").lower() in ("1", "true", "yes")

# Cap the reference to a seeded sample so the artifact stays small and drift compute is fast;
# a few-thousand-row sample is ample to characterize each feature's distribution.
DRIFT_REFERENCE_SAMPLE = int(os.getenv("DRIFT_REFERENCE_SAMPLE", "20000"))

# Dataset-level alert threshold on the share of drifted columns (0..1). Referenced by the
# serving-side gauge semantics and by observability/alert_rules.yml (keep the two in sync).
DRIFT_ALERT_SHARE = float(os.getenv("DRIFT_ALERT_SHARE", "0.3"))

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
