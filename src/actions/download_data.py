from __future__ import annotations

import copernicusmarine as cm
import pandas as pd
from pathlib import Path
from typing import Optional, Sequence
from src.config import OUTPUT_ROOT, join_uri
from src.copernicus.cm_credentials import CMCredentials
from src.copernicus.cm_subset_client import CMSubsetClient
from src.data_pulling import BBoxRowProcessor, SummaryDownloader


def main(
    *,
    cm,  # pre-authenticated Copernicus Marine client instance
    dataset_id: str,
    variables: Sequence[str],
    summary_path: str | Path,
    output_dir: str | Path,
    padding_deg: float = 0.08,
    filename_extension: str = ".nc",
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    extra_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Parameterize product details, prepare per-row jobs, and execute downloads.

    Assumes the environment is already logged in. Call `CMCredentials().ensure_present()`
    before invoking main(); it will raise early if creds are missing.
    """
    # Build thin client
    cm_client = CMSubsetClient(
        cm=cm,
        dataset_id=dataset_id,
        variables=list(variables),
        min_depth=min_depth,
        max_depth=max_depth,
        extra_kwargs=dict(extra_kwargs or {}),
    )

    # Build row processor and downloader
    row_proc = BBoxRowProcessor(
        padding_deg=padding_deg, filename_extension=filename_extension
    )
    downloader = SummaryDownloader(
        cm_client=cm_client,
        summary_path=summary_path,
        output_dir=output_dir,
        row_processor=row_proc,
    )

    manifest = downloader.run()
    # Write a manifest next to outputs
    manifest_path = Path(output_dir) / "download_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        manifest.to_csv(f, index=False)
    return manifest


if __name__ == "__main__":
    # 1) Ensure credentials exist (fails fast if not)
    CMCredentials().ensure_present()

    # 3) Configure your product + run
    #    Adjust these defaults to your project or wrap with argparse if desired.
    DATASET_ID = "C3S-GLO-SST-L4-REP-OBS-SST"
    VARIABLES = ["analysed_sst"]
    SUMMARY_PATH =  join_uri(OUTPUT_ROOT,  "date_bbox_summary_test.csv")
    OUTPUT_DIR =  join_uri(OUTPUT_ROOT, "C3S-GLO-SST-L4-REP-OBS-SST")

    # Depth: aim for first level only (example: surface layer)
    MIN_DEPTH = 0.0
    MAX_DEPTH = 2.0

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
    )
    print(
        f"Downloaded {len(manifest_df)} files. Manifest saved to: {Path(OUTPUT_DIR) / 'download_manifest.csv'}"
    )
