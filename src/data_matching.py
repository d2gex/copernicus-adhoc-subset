from __future__ import annotations
import pandas as pd
import xarray as xr

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from src import config


@dataclass(frozen=True)
class CsvColumns:
    unique_id: str
    survey: str
    lon: str
    lat: str
    time: Optional[str] = None  # optional (dd/mm/YYYY)


class NearestExtractor:
    """Per-row extractor via xarray nearest selection."""

    def __init__(
        self,
        *,
        variables: Sequence[str],
        lon_dim: str,
        lat_dim: str,
        time_dim: str = "time",
        depth_dim: str = "depth",
    ) -> None:
        self.variables = list(variables)
        self.lon_dim = lon_dim
        self.lat_dim = lat_dim
        self.time_dim = time_dim
        self.depth_dim = depth_dim

    def extract(
        self,
        *,
        ds: xr.Dataset,
        lon: float,
        lat: float,
        row_time: Optional[pd.Timestamp],
    ) -> dict:
        sel = ds.sel({self.lon_dim: lon, self.lat_dim: lat}, method="nearest")
        if (row_time is not None) and (
            self.time_dim in sel.dims or self.time_dim in sel.coords
        ):
            sel = sel.sel({self.time_dim: pd.to_datetime(row_time)}, method="nearest")
        if (self.depth_dim in sel.dims) or (self.depth_dim in sel.coords):
            sel = sel.sel({self.depth_dim: 0}, method="nearest")

        out = {}
        any_value = False
        any_missing = False
        for v in self.variables:
            if v not in sel:
                any_missing = True
                out[v] = None
                continue
            arr = sel[v]
            try:
                val = arr.values.item()
            except Exception:
                val = arr.values
            if pd.isna(val):
                out[v] = float("nan")
            else:
                any_value = True
                out[v] = val

        if any_missing:
            out["match_status"] = -1
        elif any_value:
            out["match_status"] = 1
        else:
            out["match_status"] = 0
        return out


class OriginalCsvDigester:
    """Sort by survey, keep one tile in memory, call NearestExtractor per row, accumulate results."""

    def __init__(
        self,
        *,
        original_csv: str | Path,
        tiles_dir: str | Path,
        variables: Sequence[str],
        csv_cols: CsvColumns,
        lon_dim: str,
        lat_dim: str,
        time_dim: str = "time",
        depth_dim: str = "depth",
        tile_extension: str = ".nc",
        engine: Optional[str] = None,
    ) -> None:
        self.original_csv = Path(original_csv)
        self.tiles_dir = Path(tiles_dir)
        self.variables = list(variables)
        self.csv_cols = csv_cols
        self.lon_dim = lon_dim
        self.lat_dim = lat_dim
        self.time_dim = time_dim
        self.depth_dim = depth_dim
        self.tile_extension = (
            tile_extension if tile_extension.startswith(".") else "." + tile_extension
        )
        self.engine = engine
        self.extractor = NearestExtractor(
            variables=self.variables,
            lon_dim=lon_dim,
            lat_dim=lat_dim,
            time_dim=time_dim,
            depth_dim=depth_dim,
        )

    def _open_tile(self, survey_value: str) -> xr.Dataset:
        path = self.tiles_dir / f"{survey_value}{self.tile_extension}"
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return (
            xr.open_dataset(path, engine=self.engine)
            if self.engine
            else xr.open_dataset(path)
        )

    def run(self) -> pd.DataFrame:
        with open(self.original_csv, "rb") as f:
            df = pd.read_csv(f)

        required = [
            self.csv_cols.unique_id,
            self.csv_cols.survey,
            self.csv_cols.lon,
            self.csv_cols.lat,
        ]
        for c in required:
            if c not in df.columns:
                raise KeyError(f"Missing column: {c}")
        if self.csv_cols.time is not None and self.csv_cols.time not in df.columns:
            raise KeyError(f"Missing column: {self.csv_cols.time}")

        cols = [
            self.csv_cols.unique_id,
            self.csv_cols.survey,
            self.csv_cols.lon,
            self.csv_cols.lat,
        ]
        if self.csv_cols.time:
            cols.append(self.csv_cols.time)
        work = df[cols].copy()

        if self.csv_cols.time:
            work[self.csv_cols.time] = pd.to_datetime(
                work[self.csv_cols.time],
                format="%d/%m/%Y",
                dayfirst=True,
                errors="coerce",
            )

        work = work.sort_values(self.csv_cols.survey).reset_index(drop=True)

        results: List[dict] = []
        current_survey: Optional[str] = None
        ds: Optional[xr.Dataset] = None

        try:
            for row in work.itertuples(index=False):
                survey = str(getattr(row, self.csv_cols.survey))
                if survey != current_survey:
                    if ds is not None:
                        ds.close()
                        ds = None
                    ds = self._open_tile(survey)
                    current_survey = survey

                uid = getattr(row, self.csv_cols.unique_id)
                lon = getattr(row, self.csv_cols.lon)  # already a float
                lat = getattr(row, self.csv_cols.lat)  # already a float
                row_time = (
                    getattr(row, self.csv_cols.time) if self.csv_cols.time else None
                )

                try:
                    vals = self.extractor.extract(
                        ds=ds, lon=lon, lat=lat, row_time=row_time
                    )
                except Exception:
                    raise
                    # vals = {v: None for v in self.variables}
                    # vals["match_status"] = -1

                rec = {
                    self.csv_cols.unique_id: uid,
                    self.csv_cols.survey: survey,
                    self.csv_cols.lon: lon,
                    self.csv_cols.lat: lat,
                }
                if self.csv_cols.time:
                    rec[self.csv_cols.time] = row_time
                rec.update(vals)
                results.append(rec)
        finally:
            if ds is not None:
                ds.close()

        return pd.DataFrame(results)
