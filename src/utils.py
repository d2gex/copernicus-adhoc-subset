import pandas as pd
import os
from pathlib import Path
from typing import Union, Optional, Dict, Any


def _is_s3(path: Union[str, Path]) -> bool:
    return str(path).startswith("s3://")

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


def write_csv(
    df: pd.DataFrame,
    path: Union[str, Path],
    storage_options: Optional[Dict[str, Any]] = None,
    **to_csv_kwargs,
) -> None:
    """Write CSV to local or S3 path transparently."""
    if _is_s3(path):
        opts = {"anon": False}
        if storage_options:
            opts.update(storage_options)
        df.to_csv(str(path), index=False, storage_options=opts, **to_csv_kwargs)
    else:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False, **to_csv_kwargs)


def read_csv(
    path: Union[str, Path],
    storage_options: Optional[Dict[str, Any]] = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Read CSV from local or S3 path transparently."""
    if _is_s3(path):
        opts = {"anon": False}
        if storage_options:
            opts.update(storage_options)
        return pd.read_csv(str(path), storage_options=opts, **read_csv_kwargs)
    return pd.read_csv(Path(path), **read_csv_kwargs)
