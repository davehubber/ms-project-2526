import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
NETWORKS_DIR = DATA_DIR / "networks"
SENSOR_DATA_DIR = DATA_DIR / "sensor_data"
ROUTES_DIR = DATA_DIR / "routes"
GENERATED_DIR = DATA_DIR / "generated"
MODELS_DIR = DATA_DIR / "models"

RESULTS_DIR = BASE_DIR / "results"

NET_FILE = str(NETWORKS_DIR / "net.net.xml")
TAZ_FILE = str(NETWORKS_DIR / "taz.xml")
DETECTORS_FILE = str(NETWORKS_DIR / "detectors_updated.xml") 
DETECTORS_ADD_FILE = str(NETWORKS_DIR / "detectors.add.xml")

GROUND_TRUTH_FILE = str(SENSOR_DATA_DIR / "ground_truth.csv")
SENSORS_LOCATION_FILE = str(SENSOR_DATA_DIR / "sensors_location.csv")
SENSOR_YEARS_DIR = SENSOR_DATA_DIR / "raw_years"

DAILY_TRIPS_FILE = str(ROUTES_DIR / "calibrated_daily_trips.rou.xml")
TRIPS_FILE = str(ROUTES_DIR / "trips.trips.xml")
TAZ_DUMMY_TRIPS_FILE = str(ROUTES_DIR / "taz_dummy_trips.xml")

BASELINE_MATRIX_FILE = str(GENERATED_DIR / "baseline_matrix.xml")
DATASET_FILE = str(GENERATED_DIR / "dataset.csv")
COUNTS_FILE = str(GENERATED_DIR / "counts.xml")
INITIAL_OD_FILE = str(GENERATED_DIR / "initial_od_vector.json")

METAMODEL_FILE = str(MODELS_DIR / "metamodel.pth")
SCALER_X_FILE = str(MODELS_DIR / "scaler_x.pkl")
SCALER_Y_FILE = str(MODELS_DIR / "scaler_y.pkl")
