from pathlib import Path
from typing import Union, Optional, Dict, Any
import pandas as pd


def _is_s3(path: Union[str, Path]) -> bool:
    return str(path).startswith("s3://")


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
