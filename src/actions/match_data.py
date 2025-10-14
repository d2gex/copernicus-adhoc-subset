from __future__ import annotations

from pathlib import Path
from typing import Union

from src import config
from src.utils import _is_s3, join_uri, write_csv
from src.data_matching import CsvColumns, OriginalCsvDigester


# -------------------------
# Main (local or S3-ready)
# -------------------------
def main() -> None:
    # Root where points CSV and tiles live (S3 or local). config.OUTPUT_ROOT may be Path or s3://
    root_folder: Union[str, Path] = config.OUTPUT_ROOT

    ORIGINAL_CSV = join_uri(root_folder, "points_with_index.csv")
    TILES_DIR    = join_uri(root_folder, "C3S-GLO-SST-L4-REP-OBS-SST")

    VARIABLES = ["analysed_sst"]  # example variable
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
        engine=None,  # local: auto/your choice; S3: defaults to h5netcdf internally
    )

    out = digester.run()

    # Save result next to inputs (S3 or local) using your utils
    out_csv = join_uri(root_folder, "C3S-GLO-SST-L4-REP-OBS-SST.csv") if _is_s3(root_folder) \
              else Path(root_folder) / "C3S-GLO-SST-L4-REP-OBS-SST.csv"
    write_csv(out, out_csv, index=False)


if __name__ == "__main__":
    main()

