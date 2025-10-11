import os
from pathlib import Path
from typing import Union
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root if present

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

# --- env -> config ---
S3_BUCKET = _required("S3_BUCKET")
S3_OUTPUT_PREFIX = _required("S3_OUTPUT_PREFIX")
S3_INPUT_PREFIX = _optional("S3_INPUT_PREFIX", "input")

USE_S3_INPUT = _as_bool(os.getenv("USE_S3_INPUT"), False)

LOCAL_INPUT_PATH = Path(_optional("LOCAL_INPUT_PATH", "/tmp/input"))
LOCAL_OUTPUT_PATH = Path(_optional("LOCAL_OUTPUT_PATH", "/tmp/output"))

# Roots
OUTPUT_ROOT = s3_join(S3_BUCKET, S3_OUTPUT_PREFIX)
INPUT_ROOT = s3_join(S3_BUCKET, S3_INPUT_PREFIX) if USE_S3_INPUT else LOCAL_INPUT_PATH

