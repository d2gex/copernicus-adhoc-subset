from __future__ import annotations
import csv
import math
import pandas as pd

from dataclasses import dataclass
from typing import List, Optional, Tuple
from src import config


# --- Results container ---
@dataclass(frozen=True)
class BBoxResult:
    survey_number: object
    date: object
    date_first: object
    date_last: object
    date_unique_count: int
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    avg_lat: float
    width_deg: float
    height_deg: float
    area_deg2: float
    width_km: float
    height_km: float
    area_km2: float
    diagonal_km: float


# --- Class 1: bounding-box calculations ---
class BoundingBoxBuilder:
    """Axis-aligned bbox with cosine(longitude) adjustment. Pads single-point bboxes."""

    KM_PER_DEG_LAT: float = 111.32

    def __init__(self, *, padding_deg: float = 0.08) -> None:
        self.padding_deg = float(padding_deg)

    def build_for_coords(
        self, lats: pd.Series, lons: pd.Series
    ) -> Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]:
        if lats.empty or lons.empty:
            raise ValueError("Empty coordinate series")

        lat_num = pd.to_numeric(lats, errors="coerce")
        lon_num = pd.to_numeric(lons, errors="coerce")
        if lat_num.isna().all() or lon_num.isna().all():
            raise ValueError("All coordinates are NaN after coercion")

        min_lat = float(lat_num.min())
        max_lat = float(lat_num.max())
        min_lon = float(lon_num.min())
        max_lon = float(lon_num.max())

        height_deg = max_lat - min_lat
        width_deg = max_lon - min_lon

        # Pad zero-area bbox (single point) using the same padding knob
        if width_deg == 0.0 and height_deg == 0.0:
            pad = self.padding_deg
            min_lon = max(-180.0, min_lon - pad)
            max_lon = min(180.0, max_lon + pad)
            min_lat = max(-90.0, min_lat - pad)
            max_lat = min(90.0, max_lat + pad)
            height_deg = max_lat - min_lat
            width_deg = max_lon - min_lon

        avg_lat = (min_lat + max_lat) / 2.0

        lat_rad = math.radians(avg_lat)
        width_km = abs(width_deg) * self.KM_PER_DEG_LAT * math.cos(lat_rad)
        height_km = abs(height_deg) * self.KM_PER_DEG_LAT

        area_deg2 = abs(width_deg) * abs(height_deg)
        area_km2 = width_km * height_km
        diagonal_km = math.hypot(width_km, height_km)

        return (
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            avg_lat,
            width_deg,
            height_deg,
            area_deg2,
            width_km,
            height_km,
            area_km2,
            diagonal_km,
        )


# --- Class 2: dataset digestion ---
class DatasetDigest:
    """Load → sanitize lon/lat → add row index → per-survey bbox → add avg_lat → write."""

    def __init__(
        self,
        input_path: str,
        output_points_path: str,
        output_summary_path: str,
        bbox_builder: BoundingBoxBuilder,
        lat_column_candidates: Optional[List[str]] = None,
        lon_column_candidates: Optional[List[str]] = None,
        date_column_candidates: Optional[List[str]] = None,
        survey_column_candidates: Optional[List[str]] = None,
    ) -> None:
        self.input_path = input_path
        self.output_points_path = output_points_path
        self.output_summary_path = output_summary_path
        self.bbox_builder = bbox_builder
        self.lat_column_candidates = lat_column_candidates or ["lat", "latitude", "y"]
        self.lon_column_candidates = lon_column_candidates or [
            "lon",
            "lng",
            "longitude",
            "x",
        ]
        self.date_column_candidates = date_column_candidates or [
            "date",
            "datetime",
            "time",
            "timestamp",
        ]
        self.survey_column_candidates = survey_column_candidates or [
            "survey_number",
            "survey",
            "survey_id",
            "transect_id",
        ]

    # --- internals ---
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> str:
        lowered = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in lowered:
                return lowered[cand]
        raise KeyError(f"Missing any of columns {candidates} in {list(df.columns)}")

    def load(self) -> pd.DataFrame:
        encodings = ["utf-8-sig", "utf-8", "latin-1"]
        last_err: Optional[Exception] = None
        for enc in encodings:
            try:
                with open(self.input_path, "r", encoding=enc, newline="") as f:
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
                    raise ValueError("Input CSV is empty")
                return df
            except Exception as e:
                last_err = e
        for enc in encodings:
            try:
                with open(self.input_path, "r", encoding=enc, newline="") as f:
                    df = pd.read_csv(f, engine="python", on_bad_lines="skip")
                if df.empty:
                    raise ValueError("Input CSV is empty after skipping bad lines")
                return df
            except Exception as e:
                last_err = e
        if last_err:
            raise last_err
        raise RuntimeError("Unknown CSV loading failure")

    def add_unique_index(
        self, df: pd.DataFrame, index_col: str = "row_id"
    ) -> pd.DataFrame:
        if index_col in df.columns:
            raise ValueError(f"Column '{index_col}' already exists")
        out = df.copy()
        out[index_col] = range(1, len(out) + 1)
        return out

    def _sanitize_lon_lat_inplace(self, df: pd.DataFrame) -> Tuple[str, str]:
        lat_col = self._find_column(df, self.lat_column_candidates)
        lon_col = self._find_column(df, self.lon_column_candidates)

        # Normalize odd characters and force numeric
        for col in (lat_col, lon_col):
            if df[col].dtype == object:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .str.replace("\u00a0", "", regex=False)
                    .str.replace(r"\s+", "", regex=True)
                    .str.strip()
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        bad = df[lon_col].isna() | df[lat_col].isna()
        if bad.any():
            # Keep strict: fail fast and show a few offenders by index
            bad_idx = df.index[bad][:5].tolist()
            raise ValueError(
                f"Non-numeric lon/lat after sanitization; example row indices: {bad_idx}"
            )
        return lat_col, lon_col

    def _normalized_view(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, str, str, str, str]:
        # df already sanitized in run(); this view just selects columns and re-validates numerics
        lat_col = self._find_column(df, self.lat_column_candidates)
        lon_col = self._find_column(df, self.lon_column_candidates)
        date_col = self._find_column(df, self.date_column_candidates)
        survey_col = self._find_column(df, self.survey_column_candidates)
        work = df[[lat_col, lon_col, date_col, survey_col]].copy()
        # Defensive: ensure numeric dtype (should already be float from sanitization)
        work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
        work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
        work = work.dropna(subset=[lat_col, lon_col, date_col, survey_col])
        return work, lat_col, lon_col, date_col, survey_col

    def compute_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        work, lat_col, lon_col, date_col, survey_col = self._normalized_view(df)
        rows: List[BBoxResult] = []
        for survey_value, grp in work.groupby(survey_col):
            date_first = grp[date_col].min()
            date_last = grp[date_col].max()
            date_unique_count = int(grp[date_col].nunique())
            (
                min_lon,
                min_lat,
                max_lon,
                max_lat,
                avg_lat,
                width_deg,
                height_deg,
                area_deg2,
                width_km,
                height_km,
                area_km2,
                diagonal_km,
            ) = self.bbox_builder.build_for_coords(grp[lat_col], grp[lon_col])
            rows.append(
                BBoxResult(
                    survey_number=survey_value,
                    date=date_first,
                    date_first=date_first,
                    date_last=date_last,
                    date_unique_count=date_unique_count,
                    min_lon=min_lon,
                    min_lat=min_lat,
                    max_lon=max_lon,
                    max_lat=max_lat,
                    avg_lat=avg_lat,
                    width_deg=width_deg,
                    height_deg=height_deg,
                    area_deg2=area_deg2,
                    width_km=width_km,
                    height_km=height_km,
                    area_km2=area_km2,
                    diagonal_km=diagonal_km,
                )
            )
        summary = pd.DataFrame([r.__dict__ for r in rows])

        # Assert: no zero-area bboxes remain
        zeros = summary.index[summary["area_deg2"] == 0.0].tolist()
        assert len(zeros) == 0, f"Zero-area bboxes generated for rows: {zeros}"

        return summary.sort_values("area_km2", ascending=False).reset_index(drop=True)

    def add_avg_lat_to_points(
        self, df: pd.DataFrame, summary: pd.DataFrame
    ) -> pd.DataFrame:
        survey_col = self._find_column(df, self.survey_column_candidates)
        out = df.copy()
        out["avg_lat"] = out[survey_col].map(
            dict(zip(summary["survey_number"], summary["avg_lat"]))
        )
        return out

    def write(
        self, df_points: pd.DataFrame, df_summary: pd.DataFrame
    ) -> Tuple[str, str]:
        with open(self.output_points_path, "w", newline="") as f_out:
            df_points.to_csv(f_out, index=False)
        with open(self.output_summary_path, "w", newline="") as f_out:
            df_summary.to_csv(f_out, index=False)
        return self.output_points_path, self.output_summary_path

    # --- public API ---
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = self.load()

        # Sanitize lon/lat in the full dataset (affects both outputs)
        self._sanitize_lon_lat_inplace(df)

        df_with_index = self.add_unique_index(df)
        summary = self.compute_summary(df_with_index)
        df_with_avg_lat = self.add_avg_lat_to_points(df_with_index, summary)
        self.write(df_with_avg_lat, summary)
        return df_with_avg_lat, summary
