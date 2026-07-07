"""
config.py
---------
Central place for default paths, column groupings, and thresholds used
across the claim-denial-prediction pipeline. Most of these can be
overridden via CLI flags in train.py / evaluate.py / predict.py; the
values here are just the defaults.
"""

# ---------------------------------------------------------------------
# Default file paths (override with --data_path / --model_path / etc.)
# ---------------------------------------------------------------------
DEFAULT_CLAIMS_HISTORY_PATH = "data/claims_history.csv"
DEFAULT_CURRENT_CLAIMS_PATH = "data/current_claims.csv"
DEFAULT_MODEL_PATH = "outputs/model.pkl"
DEFAULT_OUTPUT_DIR = "outputs"

# ---------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------
TARGET_COL = "is_denied"

ID_COLS = ["claim_id", "payer_id", "split"]

BOOLEAN_COLS = [
    "prior_auth_required",
    "has_prior_auth",
    "is_in_network",
    "missing_documentation_flag",
    "eligibility_verified",
    "referral_required",
    "referral_present",
    "is_denied",
]

NUMERICAL_COLS = [
    "total_billed",
    "expected_payment",
    "num_procedures",
    "num_diagnoses",
    "days_to_submit",
]

CATEGORICAL_COLS = ["payer_type", "visit_type", "service_month"]

PASSTHROUGH_COLS = [
    "prior_auth_required",
    "has_prior_auth",
    "is_in_network",
    "missing_documentation_flag",
    "eligibility_verified",
    "referral_required",
    "referral_present",
]

IGNORE_COLS_FOR_MODEL = ["claim_id", "payer_id", "split", "is_denied"]

# ---------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------
DEFAULT_MODEL_TYPE = "logreg"
RANDOM_STATE = 42
MAX_ITER = 1000
CLASS_WEIGHT = "balanced"

# ---------------------------------------------------------------------
# Risk tier thresholds
# ---------------------------------------------------------------------
HIGH_RISK_THRESHOLD = 0.85
MEDIUM_RISK_THRESHOLD = 0.50

# ---------------------------------------------------------------------
# Bucketing (for precision/recall-by-score-band analysis)
# ---------------------------------------------------------------------
NUM_SCORE_BUCKETS = 20
