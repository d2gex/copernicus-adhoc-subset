import os
from pathlib import Path
from typing import Union
from dotenv import load_dotenv

load_dotenv()

def _as_bool(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise ValueError(f"Missing required env var: {name}")
    return v


def _optional(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default


def s3_join(bucket: str, *parts: str) -> str:
    left = f"s3://{bucket}"
    right = "/".join(p.strip("/") for p in parts)
    return f"{left}/{right}"


def join_uri(base: Union[str, Path], *parts: str) -> Union[str, Path]:
    if isinstance(base, str) and base.startswith("s3://"):
        left = base.rstrip("/")
        right = "/".join(p.strip("/") for p in parts)
        return f"{left}/{right}"
    return Path(base, *parts)


file_path = Path(__file__).resolve()
ROOT_PATH = file_path.parents[1]
LOCAL_DATA_PATH = ROOT_PATH.parent / "data"
S3_BUCKET = _required("S3_BUCKET")
S3_OUTPUT_PREFIX = _required("S3_OUTPUT_PREFIX")
USE_S3_INPUT = _as_bool(os.getenv("USE_S3_INPUT"), False)

# Roots
OUTPUT_ROOT = s3_join(S3_BUCKET, S3_OUTPUT_PREFIX)
INPUT_ROOT = LOCAL_DATA_PATH / "cetaceans"
