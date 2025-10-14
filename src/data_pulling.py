from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, List, Tuple, Union
import math
import tempfile

import boto3
import pandas as pd

from src.utils import _is_s3, join_uri, read_csv  # reuse your helpers
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
    date_first: str = "date_first"
    date_last: str = "date_last"


@dataclass(frozen=True)
class SubsetJob:
    survey_number: str
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    start_datetime: Optional[str]
    end_datetime: Optional[str]
    output_filename: str


# -------------------------------
# Class 1: prepare jobs per row
# -------------------------------
class BBoxRowProcessor:
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

    def _as_date_str(self, value) -> Optional[str]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
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
        min_lon = max(-180.0, min_lon)
        max_lon = min(180.0, max_lon)
        min_lat = max(-90.0, min_lat)
        max_lat = min(90.0, max_lat)
        if min_lon > max_lon:
            min_lon, max_lon = max_lon, min_lon
        if min_lat > max_lat:
            min_lat, max_lat = max_lat, min_lat
        return (min_lon, max_lon, min_lat, max_lat)

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
    Load a summary CSV, build SubsetJob per row, then download via CMSubsetClient.
    - If output_dir is local: write directly there.
    - If output_dir is s3://… : write to a local temp dir, then upload each file to S3.
    """

    def __init__(
        self,
        cm_client: CMSubsetClient,
        *,
        summary_path: str | Path,
        output_dir: str | Path,
        row_processor: BBoxRowProcessor,
        columns: Optional[BBoxColumns] = None,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
    ) -> None:
        self.client = cm_client
        # keep S3 as str; local as Path
        self.summary_path: Union[str, Path] = (
            str(summary_path) if _is_s3(summary_path) else Path(summary_path)
        )
        self.output_dir: Union[str, Path] = (
            str(output_dir) if _is_s3(output_dir) else Path(output_dir)
        )
        self.row_processor = row_processor
        self.cols = columns or row_processor.cols
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime

    def _load_summary(self) -> pd.DataFrame:
        # Use your utils.read_csv; allow pandas to infer delimiter (python engine).
        # First pass: strict; second: tolerant.
        try:
            df = read_csv(
                self.summary_path, engine="python", sep=None, on_bad_lines="error"
            )
        except Exception:
            df = read_csv(
                self.summary_path, engine="python", sep=None, on_bad_lines="skip"
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

    def _parse_s3(self, uri: str) -> tuple[str, str]:
        # s3://bucket/prefix -> (bucket, prefix)
        tail = uri[5:]
        bucket, _, key = tail.partition("/")
        if not bucket or not key:
            raise ValueError(f"Bad S3 URI: {uri}")
        return bucket, key

    def run(self) -> pd.DataFrame:
        writing_to_s3 = _is_s3(self.output_dir)

        if writing_to_s3:
            tmp = tempfile.TemporaryDirectory()
            local_out = Path(tmp.name)
        else:
            local_out = Path(self.output_dir)
            local_out.mkdir(parents=True, exist_ok=True)

        df = self._load_summary()

        jobs: List[SubsetJob] = [
            self.row_processor.prepare_job(row) for _, row in df.iterrows()
        ]

        for job in jobs:
            self.client.subset_one(
                bbox=(job.min_lon, job.max_lon, job.min_lat, job.max_lat),
                output_filename=job.output_filename,
                output_directory=local_out,  # always local filesystem for the CM client
                start_datetime=job.start_datetime or self.start_datetime,
                end_datetime=job.end_datetime or self.end_datetime,
            )

        if writing_to_s3:
            s3 = boto3.client("s3")
            bucket, prefix = self._parse_s3(str(self.output_dir))
            for j in jobs:
                src = local_out / j.output_filename
                key = "/".join([prefix.rstrip("/"), j.output_filename])
                s3.upload_file(str(src), bucket, key)

        manifest_rows = []
        for j in jobs:
            if writing_to_s3:
                out_path = join_uri(
                    self.output_dir, j.output_filename
                )  # s3://bucket/prefix/file
            else:
                out_path = str(Path(self.output_dir) / j.output_filename)
            manifest_rows.append(
                {
                    "survey_number": j.survey_number,
                    "min_lon": j.min_lon,
                    "max_lon": j.max_lon,
                    "min_lat": j.min_lat,
                    "max_lat": j.max_lat,
                    "start_datetime": j.start_datetime,
                    "end_datetime": j.end_datetime,
                    "output_path": out_path,
                }
            )

        return pd.DataFrame(manifest_rows)
