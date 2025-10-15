from __future__ import annotations

import math
import tempfile
import boto3
import pandas as pd
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, List, Tuple, Union

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

        # normalize constructor dates once
        self._start_iso = self._parse_ddmmyyyy_to_iso(self.start_datetime)
        self._end_iso = self._parse_ddmmyyyy_to_iso(self.end_datetime)

    # add inside class SummaryDownloader
    def _parse_ddmmyyyy_to_iso(self, value: Optional[str]) -> Optional[str]:
        """dd/mm/YYYY -> 'YYYY-MM-DD', or None if empty/invalid."""
        if value is None:
            return None
        ts = pd.to_datetime(value, format="%d/%m/%Y", dayfirst=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date().isoformat()

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

    def _zip_dir(self, src_dir: Path, zip_path: Path) -> None:
        # Build a stable list first and exclude the zip we're creating
        files = [
            p for p in sorted(src_dir.iterdir())
            if p.is_file() and p.resolve() != zip_path.resolve()
        ]
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in files:
                zf.write(p, arcname=p.name)

    def _create_zip_bundle(self, local_out: Path) -> tuple[str, Path]:
        """
        Build zip name from the last segment of output_dir, zip local_out, return (zip_name, zip_path).
        """
        if _is_s3(self.output_dir):
            _, prefix = self._parse_s3(str(self.output_dir))
            folder_basename = prefix.rstrip("/").split("/")[-1]
        else:
            folder_basename = Path(self.output_dir).name

        zip_name = f"{folder_basename}.zip"
        zip_path = local_out / zip_name
        self._zip_dir(local_out, zip_path)
        return zip_name, zip_path

    def _upload_zip_bundle_to_s3(
        self, *, s3, bucket: str, prefix: str, zip_name: str, zip_path: Path
    ) -> None:
        """
        Upload the previously created zip bundle to s3://bucket/prefix/zip_name.
        """
        zip_key = "/".join([prefix.rstrip("/"), zip_name])
        s3.upload_file(str(zip_path), bucket, zip_key)

    def _constrain_dates(
        self,
        row_start_iso: Optional[str],
        row_end_iso: Optional[str],
    ) -> tuple[Optional[str], Optional[str], bool]:
        """
        Intersect row dates with global dates.
        If BOTH global dates are None, skip constraining entirely and never invalidate.
        Returns (req_start_iso, req_end_iso, valid_interval).
        """
        # No global constraints: pass row dates through and mark as valid.
        if self._start_iso is None and self._end_iso is None:
            return row_start_iso, row_end_iso, True

        starts = [d for d in (row_start_iso, self._start_iso) if d is not None]
        ends = [d for d in (row_end_iso, self._end_iso) if d is not None]
        req_start = max(starts) if starts else None
        req_end = min(ends) if ends else None

        # Only invalidate when both bounds exist AND the intersection is empty.
        valid = not (
            req_start is not None and req_end is not None and req_start > req_end
        )
        return req_start, req_end, valid

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

        # Execute only valid (date-constrained) jobs; keep constrained dates for manifest.
        executed: List[tuple[SubsetJob, Optional[str], Optional[str]]] = []

        for job in jobs:
            req_start, req_end, valid = self._constrain_dates(
                job.start_datetime, job.end_datetime
            )
            if not valid:
                print(
                    f"[skip] survey={job.survey_number} dates out of range ({req_start}..{req_end})"
                )
            else:
                self.client.subset_one(
                    bbox=(job.min_lon, job.max_lon, job.min_lat, job.max_lat),
                    output_filename=job.output_filename,
                    output_directory=local_out,  # always local filesystem for the CM client
                    start_datetime=req_start,
                    end_datetime=req_end,
                )
                executed.append((job, req_start, req_end))

        # Create zip bundle from local_out (unchanged behavior)
        zip_name, zip_path = self._create_zip_bundle(local_out)

        if writing_to_s3:
            s3 = boto3.client("s3")
            bucket, prefix = self._parse_s3(str(self.output_dir))

            # Upload only files that were actually executed
            for job, _, _ in executed:
                src = local_out / job.output_filename
                key = "/".join([prefix.rstrip("/"), job.output_filename])
                s3.upload_file(str(src), bucket, key)

            # Upload the zip bundle
            self._upload_zip_bundle_to_s3(
                s3=s3,
                bucket=bucket,
                prefix=prefix,
                zip_name=zip_name,
                zip_path=zip_path,
            )

        # Manifest reflects only executed jobs, with constrained dates
        manifest_rows: List[dict] = []
        for job, req_start, req_end in executed:
            if writing_to_s3:
                out_path = join_uri(
                    self.output_dir, job.output_filename
                )  # s3://bucket/prefix/file
            else:
                out_path = str(Path(self.output_dir) / job.output_filename)
            manifest_rows.append(
                {
                    "survey_number": job.survey_number,
                    "min_lon": job.min_lon,
                    "max_lon": job.max_lon,
                    "min_lat": job.min_lat,
                    "max_lat": job.max_lat,
                    "start_datetime": req_start,  # constrained
                    "end_datetime": req_end,  # constrained
                    "output_path": out_path,
                }
            )

        return pd.DataFrame(manifest_rows)
