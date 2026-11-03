from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get('TAXI_DATA_DIR', str(ROOT / 'data')))
ARTIFACTS = Path(os.environ.get('TAXI_ARTIFACT_DIR', str(ROOT / 'artifacts')))
DATABASE = DATA / 'taxi.sqlite3'
SEED = 42
FEATURES = [
    'zone_id', 'borough', 'dow', 'month', 'day_of_month', 'is_weekend',
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'lag_1', 'lag_7',
    'lag_14', 'lag_28', 'rolling_mean_7', 'rolling_mean_28',
    'rolling_std_7', 'rolling_min_7', 'temperature_lag1',
    'precipitation_lag1', 'wind_lag1'
]
CATEGORICAL = ['zone_id', 'borough']
NUMERICAL = [x for x in FEATURES if x not in CATEGORICAL]
MIN_DAYS = 110

def ensure_paths():
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / 'raw').mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
