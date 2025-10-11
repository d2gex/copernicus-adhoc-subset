from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, List, Tuple
import math
import csv
import sys
import copernicusmarine as cm

import pandas as pd

from src import config
from src.copernicus.cm_credentials import CMCredentials
from src.copernicus.cm_subset_client import CMSubsetClient


# ------------------------
# Data containers & config
# ------------------------
@dataclass(frozen=True)
class BBoxColumns:
    survey: str = "survey_number"
    min_lon: str = "min_lon"
    min_lat: str = "min_lat"
    max_lon: str = "max_lon"
    max_lat: str = "max_lat"
    date_first: str = "date_first"  # optional, used for start_datetime
    date_last: str = "date_last"  # optional, used for end_datetime


@dataclass(frozen=True)
class SubsetJob:
    """All inputs required to call CMSubsetClient.subset_one for one row."""

    survey_number: str
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    start_datetime: Optional[str]  # YYYY-MM-DD or None
    end_datetime: Optional[str]  # YYYY-MM-DD or None
    output_filename: str  # e.g., "<survey_number>.nc"


# -------------------------------
# Class 1: prepare jobs per row
# -------------------------------
class BBoxRowProcessor:
    """
    Transform a single summary-row into a SubsetJob (no network calls).

    - Applies symmetric padding (degrees) to lon/lat, clamps to valid ranges.
    - Accepts configurable column names via BBoxColumns.
    - Converts date fields to 'YYYY-MM-DD' if present; otherwise None.
    - Names the output strictly from survey_number + extension.
    """

    def __init__(
        self,
        *,
        padding_deg: float = 0.08,
        filename_extension: str = ".nc",
        columns: Optional[BBoxColumns] = None,
    ) -> None:
        self.padding_deg = float(padding_deg)
        self.filename_extension = (
            filename_extension
            if filename_extension.startswith(".")
            else "." + filename_extension
        )
        self.cols = columns or BBoxColumns()

    # ---- helpers ----
    def _as_date_str(self, value) -> Optional[str]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        # Explicit dd/mm/YYYY → ISO conversion
        ts = pd.to_datetime(value, format="%d/%m/%Y", dayfirst=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date().isoformat()

    def _to_num(self, x) -> float:
        if isinstance(x, str):
            x = x.replace(",", ".").strip()
        return float(x)

    def _padded_bbox(self, row: pd.Series) -> Tuple[float, float, float, float]:
        min_lon = self._to_num(row[self.cols.min_lon]) - self.padding_deg
        max_lon = self._to_num(row[self.cols.max_lon]) + self.padding_deg
        min_lat = self._to_num(row[self.cols.min_lat]) - self.padding_deg
        max_lat = self._to_num(row[self.cols.max_lat]) + self.padding_deg

        # clamp
        min_lon = max(-180.0, min_lon)
        max_lon = min(180.0, max_lon)
        min_lat = max(-90.0, min_lat)
        max_lat = min(90.0, max_lat)

        # order
        if min_lon > max_lon:
            min_lon, max_lon = max_lon, min_lon
        if min_lat > max_lat:
            min_lat, max_lat = max_lat, min_lat
        return (min_lon, max_lon, min_lat, max_lat)

    # ---- public API ----
    def prepare_job(self, row: pd.Series) -> SubsetJob:
        survey_value = str(row[self.cols.survey])
        min_lon, max_lon, min_lat, max_lat = self._padded_bbox(row)
        start_dt = self._as_date_str(row.get(self.cols.date_first))
        end_dt = self._as_date_str(row.get(self.cols.date_last))
        output_filename = f"{survey_value}{self.filename_extension}"
        return SubsetJob(
            survey_number=survey_value,
            min_lon=min_lon,
            max_lon=max_lon,
            min_lat=min_lat,
            max_lat=max_lat,
            start_datetime=start_dt,
            end_datetime=end_dt,
            output_filename=output_filename,
        )


# ----------------------------------
# Class 2: digest the whole summary
# ----------------------------------
class SummaryDownloader:
    """
    Load a summary CSV, build SubsetJob per row using BBoxRowProcessor, then
    execute downloads via CMSubsetClient. Credentials are checked outside.
    """

    def __init__(
        self,
        cm_client: CMSubsetClient,
        *,
        summary_path: str | Path,
        output_dir: str | Path,
        row_processor: BBoxRowProcessor,
        columns: Optional[BBoxColumns] = None,
        start_datetime: Optional[str] = None,  # <— ADD
        end_datetime: Optional[str] = None,
    ) -> None:
        self.client = cm_client
        self.summary_path = Path(summary_path)
        self.output_dir = Path(output_dir)
        self.row_processor = row_processor
        self.cols = columns or row_processor.cols
        self.start_datetime = start_datetime  # <— ADD
        self.end_datetime = end_datetime  # <— ADD

    # ---- internals ----
    def _load_summary(self) -> pd.DataFrame:
        encodings = ["utf-8-sig", "utf-8", "latin-1"]
        last_err: Optional[Exception] = None
        for enc in encodings:
            try:
                with open(self.summary_path, "r", encoding=enc, newline="") as f:
                    sample = f.read(65536)
                    f.seek(0)
                    try:
                        dialect = csv.Sniffer().sniff(sample)
                        sep = dialect.delimiter
                        quote = getattr(dialect, "quotechar", '"')
                    except Exception:
                        sep, quote = None, '"'
                    df = pd.read_csv(
                        f,
                        sep=sep,
                        engine="python",
                        quotechar=quote,
                        on_bad_lines="error",
                    )
                if df.empty:
                    raise ValueError("Summary CSV is empty")
                for c in [
                    self.cols.survey,
                    self.cols.min_lon,
                    self.cols.min_lat,
                    self.cols.max_lon,
                    self.cols.max_lat,
                ]:
                    if c not in df.columns:
                        raise KeyError(f"Missing required column: {c}")
                return df
            except Exception as e:
                last_err = e
        for enc in encodings:
            try:
                with open(self.summary_path, "r", encoding=enc, newline="") as f:
                    df = pd.read_csv(f, engine="python", on_bad_lines="skip")
                if df.empty:
                    raise ValueError("Summary CSV is empty after skipping bad lines")
                return df
            except Exception as e:
                last_err = e
        if last_err:
            raise last_err
        raise RuntimeError("Unknown summary loading failure")

    # ---- public API ----
    def run(self) -> pd.DataFrame:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df = self._load_summary()

        jobs: List[SubsetJob] = []
        for _, row in df.iterrows():
            jobs.append(self.row_processor.prepare_job(row))

        # Execute downloads
        for job in jobs:
            self.client.subset_one(
                bbox=(job.min_lon, job.max_lon, job.min_lat, job.max_lat),
                output_filename=job.output_filename,
                output_directory=self.output_dir,
                start_datetime=job.start_datetime,
                end_datetime=job.end_datetime,
            )

        # Return a manifest DataFrame
        manifest = pd.DataFrame(
            [
                {
                    "survey_number": j.survey_number,
                    "min_lon": j.min_lon,
                    "max_lon": j.max_lon,
                    "min_lat": j.min_lat,
                    "max_lat": j.max_lat,
                    "start_datetime": j.start_datetime,
                    "end_datetime": j.end_datetime,
                    "output_path": str(self.output_dir / j.output_filename),
                }
                for j in jobs
            ]
        )
        return manifest
