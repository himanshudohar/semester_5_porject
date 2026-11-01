"""Central configuration for JalDrishti."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

# Real data goes here (see README, Step 6). Falls back to demo data if missing.
REAL_DATA_PATH = DATA_DIR / "wells.csv"
DEMO_DATA_PATH = DATA_DIR / "demo_wells.csv"

MODEL_PATH = MODEL_DIR / "jaldrishti_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"

# Features known BEFORE drilling (no leakage: no well_depth / water_level / yield).
NUMERIC_FEATURES = [
    "rainfall", "elevation", "slope", "dist_stream", "drainage_density",
    "ndvi", "nearby_well_success_rate", "nearby_well_avg_depth",
]
CATEGORICAL_FEATURES = ["soil", "geology", "landcover"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "label"  # 1 = usable water found under your documented rule, 0 = failed/dry

# Columns that must NEVER be used as inputs (they are only known after drilling).
LEAKY_COLUMNS = ["well_depth", "water_level", "yield_lpm"]

# Geographic validation: lat/lon grid block size in degrees (~11 km per 0.1 deg)
SPATIAL_BLOCK_DEG = 0.15
RANDOM_STATE = 42

LABEL_NAMES = {0: "Low", 1: "Medium", 2: "High"}
# Probability thresholds for Low / Medium / High potential
LOW_MAX = 0.40
HIGH_MIN = 0.65
