from __future__ import annotations
import os
import copernicusmarine as cm
import pandas as pd
from pathlib import Path
from typing import Optional, Sequence, Union

from src.config import OUTPUT_ROOT
from src.utils import _is_s3, join_uri, write_csv  # reuse your helpers
from src.copernicus.cm_credentials import CMCredentials
from src.copernicus.cm_subset_client import CMSubsetClient
from src.data_pulling import BBoxRowProcessor, SummaryDownloader


def main(
    *,
    cm,  # pre-authenticated Copernicus Marine client instance
    dataset_id: str,
    variables: Sequence[str],
    summary_path: Union[str, Path],
    output_dir: Union[str, Path],
    padding_deg: float = 0.08,
    filename_extension: str = ".nc",
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    extra_kwargs: Optional[dict] = None,
    start_datetime: Optional[str] = None,  # dd/mm/YYYY
    end_datetime: Optional[str] = None,  # dd/mm/YYYY
) -> pd.DataFrame:
    """
    Parameterize product details, prepare per-row jobs, and execute downloads.
    """
    # Thin client
    cm_client = CMSubsetClient(
        cm=cm,
        dataset_id=dataset_id,
        variables=list(variables),
        min_depth=min_depth,
        max_depth=max_depth,
        extra_kwargs=dict(extra_kwargs or {}),
    )

    # Row processor + downloader
    row_proc = BBoxRowProcessor(
        padding_deg=padding_deg, filename_extension=filename_extension
    )
    downloader = SummaryDownloader(
        cm_client=cm_client,
        summary_path=summary_path,
        output_dir=output_dir,
        row_processor=row_proc,
        start_datetime=start_datetime,  # NEW: pass-through
        end_datetime=end_datetime,  # NEW: pass-through
    )

    # Run downloads (to local or S3 per SummaryDownloader logic)
    manifest = downloader.run()

    # Build manifest path "next to outputs" without breaking s3://
    manifest_path = (
        join_uri(output_dir, "download_manifest.csv")
        if _is_s3(output_dir)
        else Path(output_dir) / "download_manifest.csv"
    )

    # Persist manifest using your utils (context-managed local write; signed S3 write)
    write_csv(manifest, manifest_path)

    return manifest


if __name__ == "__main__":
    # 1) Ensure credentials exist (fails fast if not)
    CMCredentials().ensure_present()

    # 2) Configure your product + run (tweak as needed)
    DATASET_ID = os.getenv("CM_DATASET_ID").strip()
    _vars = os.getenv("CM_VARIABLES")
    VARIABLES = [v.strip() for v in _vars.split(",") if v.strip()]

    # Min/Max depth
    MIN_DEPTH = float(os.getenv("CM_MIN_DEPTH", "").strip())
    MAX_DEPTH = float(os.getenv("CM_MAX_DEPTH", "").strip())

    # Global date constraints (inclusive), dd/mm/YYYY; blank -> None
    START_DT = os.getenv("CM_START_DATE", "").strip() or None
    END_DT = os.getenv("CM_END_DATE", "").strip() or None

    # DATA DOWNLOADING INPUT
    DATA_DOWNLOADING_INPUT = os.getenv("CM_DATA_DOWNLOADING_CSV").strip()
    SUMMARY_PATH = join_uri(OUTPUT_ROOT, DATA_DOWNLOADING_INPUT)
    OUTPUT_DIR = join_uri(OUTPUT_ROOT, DATASET_ID)

    manifest_df = main(
        cm=cm,
        dataset_id=DATASET_ID,
        variables=VARIABLES,
        summary_path=SUMMARY_PATH,
        output_dir=OUTPUT_DIR,
        padding_deg=0.05,
        filename_extension=".nc",
        min_depth=MIN_DEPTH,
        max_depth=MAX_DEPTH,
        extra_kwargs=None,
        start_datetime=START_DT,
        end_datetime=END_DT,
    )

    # Friendly summary print with correct path formatting
    final_manifest_path = (
        join_uri(OUTPUT_DIR, "download_manifest.csv")
        if _is_s3(OUTPUT_DIR)
        else str(Path(OUTPUT_DIR) / "download_manifest.csv")
    )
    print(
        f"Downloaded {len(manifest_df)} files. Manifest saved to: {final_manifest_path}"
    )
