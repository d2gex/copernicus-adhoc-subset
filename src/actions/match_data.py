from __future__ import annotations
import os
from pathlib import Path
from typing import Union

from src import config
from src.utils import _is_s3, join_uri, write_csv
from src.data_matching import CsvColumns, OriginalCsvDigester


## -------------------------
# Main (local or S3-ready)
# -------------------------
def main() -> None:
    # Root where points CSV and tiles live (S3 or local). config.OUTPUT_ROOT may be Path or s3://
    root_folder: Union[str, Path] = config.OUTPUT_ROOT

    DATASET_ID = os.getenv("CM_DATASET_ID").strip()
    _vars = os.getenv("CM_VARIABLES")
    VARIABLES = [v.strip() for v in _vars.split(",") if v.strip()]
    DATA_MATCH_INPUT = os.getenv("CM_DATA_MATCH_CSV").strip()

    ORIGINAL_CSV = join_uri(root_folder, DATA_MATCH_INPUT)
    TILES_DIR = join_uri(root_folder, DATASET_ID)

    COLS = CsvColumns(
        unique_id="row_id",
        survey="survey_number",
        lon="lon",
        lat="lat",
        time="date",  # dd/mm/YYYY
    )
    LON_DIM = "longitude"
    LAT_DIM = "latitude"

    digester = OriginalCsvDigester(
        original_csv=ORIGINAL_CSV,
        tiles_dir=TILES_DIR,
        variables=VARIABLES,
        csv_cols=COLS,
        lon_dim=LON_DIM,
        lat_dim=LAT_DIM,
        time_dim="time",
        depth_dim="depth",
        tile_extension=".nc",
        engine=None,  # local backend; S3 will use h5netcdf via file handle
    )

    out = digester.run()

    # Save result next to inputs (S3 or local) using your utils
    out_csv = (
        join_uri(root_folder, f"{DATASET_ID}.csv")
        if _is_s3(root_folder)
        else Path(root_folder) / f"{DATASET_ID}.csv"
    )
    write_csv(
        out, out_csv
    )  # don't pass index: utils.write_csv already sets index=False


if __name__ == "__main__":
    main()
