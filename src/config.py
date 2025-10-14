import os
from pathlib import Path
from dotenv import load_dotenv
from src.utils import _as_bool, _required, s3_join

load_dotenv()

file_path = Path(__file__).resolve()
ROOT_PATH = file_path.parents[1]
LOCAL_DATA_PATH = ROOT_PATH.parent / "data"
S3_BUCKET = _required("S3_BUCKET")
S3_OUTPUT_PREFIX = _required("S3_OUTPUT_PREFIX")
USE_S3_INPUT = _as_bool(os.getenv("USE_S3_INPUT"), False)

# Roots
OUTPUT_ROOT = s3_join(S3_BUCKET, S3_OUTPUT_PREFIX)
INPUT_ROOT = LOCAL_DATA_PATH / "cetaceans"
