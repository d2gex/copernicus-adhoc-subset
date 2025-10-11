from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple


class CMSubsetClient:
    """Thin wrapper around copernicusmarine.subset (assumes prior login)."""

    def __init__(
        self,
        cm,
        dataset_id: str,
        variables: Sequence[str],
        *,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
        extra_kwargs: Optional[dict] = None,
    ):
        self.dataset_id = dataset_id
        self.variables = list(variables)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.extra_kwargs = dict(extra_kwargs or {})
        self.cm = cm

    def subset_one(
        self,
        bbox: Tuple[float, float, float, float],  # (min_lon, max_lon, min_lat, max_lat)
        output_filename: str,
        output_directory: str | Path,
        start_datetime: Optional[str] = None,  # "YYYY-MM-DD"
        end_datetime: Optional[str] = None,  # "YYYY-MM-DD"
    ) -> None:
        min_lon, max_lon, min_lat, max_lat = bbox
        out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        kwargs = {
            "dataset_id": self.dataset_id,
            "variables": self.variables,
            "minimum_longitude": min_lon,
            "maximum_longitude": max_lon,
            "minimum_latitude": min_lat,
            "maximum_latitude": max_lat,
            "output_filename": output_filename,
            "output_directory": str(out_dir),
        }

        if start_datetime is not None:
            kwargs["start_datetime"] = start_datetime
        if end_datetime is not None:
            kwargs["end_datetime"] = end_datetime
        if self.min_depth is not None:
            kwargs["minimum_depth"] = self.min_depth
        if self.max_depth is not None:
            kwargs["maximum_depth"] = self.max_depth
        if self.extra_kwargs:
            kwargs.update(self.extra_kwargs)

        self.cm.subset(**kwargs)

    def subset_many(
        self,
        bboxes: Iterable[Tuple[float, float, float, float]],
        periods: Iterable[Tuple[str, str]],  # (start_datetime, end_datetime)
        output_directory: str | Path,
        filename_fn: Callable[[Tuple[float, float, float, float], str, str], str],
    ) -> None:
        for start_dt, end_dt in periods:
            for bbox in bboxes:
                fname = filename_fn(bbox, start_dt, end_dt)
                self.subset_one(
                    bbox=bbox,
                    output_filename=fname,
                    output_directory=output_directory,
                    start_datetime=start_dt,
                    end_datetime=end_dt,
                )
