from pathlib import Path

# Obtiene la ruta del directorio actual del script
ROOT_PATH = Path(__file__).resolve().parent.parent

CONFIG_PATH = ROOT_PATH / 'config'

DATA_PATH = ROOT_PATH / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"

MODELS_PATH = ROOT_PATH / "models"

ARTIFACTS_PATH = ROOT_PATH / "artifacts"

NOTEBOOKS_PATH = ROOT_PATH / "notebooks"

UTILS_PATH = ROOT_PATH / "utils"

SRC_PATH = ROOT_PATH / "src"

# Rutas específicas a archivos o subdirectorios dentro de los directorios anteriores
#EXAMPLE_DATA_FILE = DATA_DIR / "example_data.csv"
#TRAINED_MODEL_FILE = MODELS_DIR / "trained_model.pkl"
#PIPELINE_FILE = ARTIFACTS_DIR / "data_processing_pipeline.joblib"

