from __future__ import annotations

import pandas as pd
import xarray as xr
import s3fs

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union, IO

from src.utils import _is_s3, join_uri, read_csv


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
    """Sort by survey, keep one tile in memory, call NearestExtractor per row, accumulate results.

    Works with local folders or S3 prefixes for both the original CSV and the tiles dir.
    """

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
        progress_every: int = 200,  # print progress every N rows
    ) -> None:
        # Keep S3 paths as strings; local paths as Path
        self.original_csv: Union[str, Path] = (
            str(original_csv) if _is_s3(original_csv) else Path(original_csv)
        )
        self.tiles_dir: Union[str, Path] = (
            str(tiles_dir) if _is_s3(tiles_dir) else Path(tiles_dir)
        )

        self.variables = list(variables)
        self.csv_cols = csv_cols
        self.lon_dim = lon_dim
        self.lat_dim = lat_dim
        self.time_dim = time_dim
        self.depth_dim = depth_dim
        self.tile_extension = (
            tile_extension if tile_extension.startswith(".") else "." + tile_extension
        )
        self.engine = engine  # for local; S3 defaults to "h5netcdf" unless you override
        self.progress_every = int(progress_every)

        self.extractor = NearestExtractor(
            variables=self.variables,
            lon_dim=lon_dim,
            lat_dim=lat_dim,
            time_dim=time_dim,
            depth_dim=depth_dim,
        )

        # S3 fs and open handle for current dataset (kept alive while ds is open)
        self._s3fs: Optional[s3fs.S3FileSystem] = None
        self._fh: Optional[IO[bytes]] = None  # file handle for current S3 tile

        # cached current tile center for diagnostics
        self._tile_center_lon: Optional[float] = None
        self._tile_center_lat: Optional[float] = None

    def _fs(self) -> s3fs.S3FileSystem:
        if self._s3fs is None:
            self._s3fs = s3fs.S3FileSystem(anon=False)
        return self._s3fs

    def _tile_path(self, survey_value: str) -> Union[str, Path]:
        """Return the tile path for a survey, preserving S3 URIs as strings."""
        filename = f"{survey_value}{self.tile_extension}"
        if _is_s3(self.tiles_dir):
            return join_uri(self.tiles_dir, filename)  # s3://bucket/prefix/file.nc
        return Path(self.tiles_dir) / filename

    def _close_current_handle(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
        finally:
            self._fh = None

    @staticmethod
    def _midpoint(a: float, b: float) -> float:
        # handles ordered [min,max] ranges including negatives
        return (a + b) / 2.0

    def _compute_tile_center(
        self, ds: xr.Dataset
    ) -> tuple[Optional[float], Optional[float]]:
        """Compute a representative center lon/lat from the tile coords."""
        try:
            lon_vals = ds[self.lon_dim].values
            lat_vals = ds[self.lat_dim].values
        except Exception:
            return (None, None)

        try:
            lon_min = float(pd.Series(lon_vals).min())
            lon_max = float(pd.Series(lon_vals).max())
            lat_min = float(pd.Series(lat_vals).min())
            lat_max = float(pd.Series(lat_vals).max())
            # NOTE: if your tiles can cross the dateline, adapt this to unwrap longitudes.
            lon_c = self._midpoint(lon_min, lon_max)
            lat_c = self._midpoint(lat_min, lat_max)
            return (lon_c, lat_c)
        except Exception:
            return (None, None)

    def _open_tile(self, survey_value: str) -> xr.Dataset:
        # Always close previous S3 handle before opening a new one
        self._close_current_handle()

        path = self._tile_path(survey_value)

        if _is_s3(path):
            # Keep the file handle open as long as the Dataset lives
            fs = self._fs()
            s3_key = str(path)[5:]  # strip "s3://"
            self._fh = fs.open(s3_key, "rb")
            eng = self.engine or "h5netcdf"
            return xr.open_dataset(self._fh, engine=eng)

        # Local filesystem
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return xr.open_dataset(p, engine=self.engine) if self.engine else xr.open_dataset(p)

    def run(self) -> pd.DataFrame:
        # Load original CSV (local or S3) using your utils
        try:
            df = read_csv(
                self.original_csv, engine="python", sep=None, on_bad_lines="error"
            )
        except Exception:
            df = read_csv(
                self.original_csv, engine="python", sep=None, on_bad_lines="skip"
            )

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
        n = len(work)
        print(
            f"[run] rows={n}, surveys={work[self.csv_cols.survey].nunique()}",
            flush=True,
        )
        next_pct = 10  # print at 10%, 20%, ... 100%

        results: List[dict] = []
        current_survey: Optional[str] = None
        ds: Optional[xr.Dataset] = None

        try:
            for i, row in enumerate(work.itertuples(index=False), start=1):
                # progress every 10% (no stdout flood)
                pct = (i * 100) // n
                if pct >= next_pct:
                    print(f"[run] {pct}% ({i}/{n})", flush=True)
                    next_pct += 10

                survey = str(getattr(row, self.csv_cols.survey))
                if survey != current_survey:
                    # close previous dataset + S3 handle
                    if ds is not None:
                        ds.close()
                        ds = None
                    self._close_current_handle()

                    # open next tile
                    ds = self._open_tile(survey)
                    current_survey = survey

                uid = getattr(row, self.csv_cols.unique_id)
                lon = getattr(row, self.csv_cols.lon)
                lat = getattr(row, self.csv_cols.lat)
                row_time = (
                    getattr(row, self.csv_cols.time) if self.csv_cols.time else None
                )

                vals = self.extractor.extract(
                    ds=ds, lon=lon, lat=lat, row_time=row_time
                )

                rec = {
                    self.csv_cols.unique_id: uid,
                    self.csv_cols.survey: survey,
                    self.csv_cols.lon: lon,
                    self.csv_cols.lat: lat,
                    # diagnostics: tile center for this survey
                    "tile_center_lon": self._tile_center_lon,
                    "tile_center_lat": self._tile_center_lat,
                }
                if self.csv_cols.time:
                    rec[self.csv_cols.time] = row_time
                rec.update(vals)
                results.append(rec)
        finally:
            # close dataset and any lingering S3 handle
            if ds is not None:
                ds.close()
            self._close_current_handle()

        print(f"[run] done. total rows={len(results)}", flush=True)
        return pd.DataFrame(results)
