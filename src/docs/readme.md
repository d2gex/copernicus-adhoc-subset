Great questions—here’s exactly what those columns mean and how I computed them, plus what “axis-wise nearest” is in xarray.

# How I computed the distances in `verification_report_axiswise.csv` (AI-generated)

For each sampled `survey_number`:

1. **Find the “own” cell center (axis-wise nearest)**

   * Open that survey’s `.nc`.
   * Get its longitude array `LON` and latitude array `LAT`.
   * Pick the index of the longitude closest to the point’s longitude (with wraparound at ±180°), and independently pick the index of the latitude closest to the point’s latitude.

     * `i_lon = argmin(|Δlon_wrapped|)`, `i_lat = argmin(|Δlat|)`
   * Cell center = `(lon_c, lat_c) = (LON[i_lon], LAT[i_lat])` (or the nearest 2-D center for curvilinear grids; see note below).

2. **Compute `own_deg_distance` (Euclidean in degree space)**

   * Compute wrapped longitude difference:
     `Δlon_wrapped = ((lon_c − point_lon + 180) % 360) − 180`
   * Latitude difference: `Δlat = lat_c − point_lat`
   * Euclidean (in **degrees**, not meters):
     `own_deg_distance = hypot(Δlon_wrapped, Δlat) = sqrt(Δlon_wrapped² + Δlat²)`

3. **Find the closest tile across neighbors**

   * For each *neighboring* survey (filtered by bbox around the point), repeat step (1) to get that tile’s axis-wise nearest cell center `(lon_c_other, lat_c_other)`.
   * Compute the same Euclidean-in-degrees distance to the point:
     `deg_distance_other = hypot(Δlon_wrapped_other, Δlat_other)`
   * Take the minimum across all neighbors (including the “own” file):

     * `closest_deg_distance = min(deg_distance_other over candidates)`
     * `closest_file_survey_number` is the survey achieving that minimum.
   * **PASS** if the minimum comes from the “own” file (so `closest_file_survey_number == survey_number`).

So:

* `own_deg_distance` = degree-space Euclidean distance from the point to the axis-wise nearest cell in its **own** `.nc`.
* `closest_deg_distance` = the **smallest** degree-space Euclidean distance among the point and all candidate tiles’ axis-wise nearest cells.

# What “axis-wise nearest” means (xarray behavior)

Yes—your understanding is right for regular (1-D) lon/lat coordinates:

* `ds.sel({lon: point_lon, lat: point_lat}, method="nearest")` finds:

  * the **lon index** whose `lon` value is closest to `point_lon`;
  * **independently**, the **lat index** whose `lat` value is closest to `point_lat`;
* then returns the data at that **(lon_index, lat_index)** pair.

It does **not** compute a single 2-D distance across all grid points. It’s two separate 1-D nearest operations (one along the longitude axis, one along the latitude axis) combined into a single index pair.

### Curvilinear note

If a dataset has **2-D** lon/lat (curvilinear grids), xarray’s `.sel` can behave differently depending on CF metadata and backends. In my axis-wise re-run, when I detected 2-D lon/lat I approximated by choosing the grid center whose `(lon,lat)` in degrees minimized `(Δlon_wrapped² + Δlat²)`—i.e., the closest **center** in degree space—then built a local rectangle from neighbor midpoints. For regular (1-D) grids, it’s pure axis-wise as described above.

# Tiny example (regular grid)

* Grid: `lon = [-10, -9.95, -9.90, ...]`, `lat = [35.00, 35.05, 35.10, ...]`
* Point: `(-9.93, 35.07)`
* Axis-wise nearest:

  * nearest lon to −9.93 is −9.95 (Δlon = −0.02)
  * nearest lat to 35.07 is 35.05 (Δlat = −0.02)
  * selected cell center = (−9.95, 35.05)
* Degree-space distance: `hypot(0.02, 0.02) ≈ 0.0283°`

You’re right to pause on terminology—that’s where we’ve been talking past each other.

* In **your wording**, “tile” = the **grid cell** returned by `.sel(..., method="nearest")`.
* In some of my earlier replies, I loosely said “tile” when I meant the **file/patch for a `survey_number`**. I’ll stop doing that.

Let’s align terms:

* **File/patch**: one `.nc` per `survey_number` (may cover a large bbox and thus contain *many* grid cells).
* **Cell (“tile” in your sense)**: the single grid cell that `.sel(..., method="nearest")` picks inside that file.

With that clarified, here’s what I did in the axis-wise re-run:

1. **`own_deg_distance`**

   * Open the **single `.nc` file** for that `survey_number`.
   * Do **one** xarray-style nearest selection conceptually (axis-wise): find the nearest **lon index** and, independently, the nearest **lat index**; that yields **one cell**.
   * Compute Euclidean distance in degree space between the input point and **that cell’s center**.

2. **`closest_deg_distance`**

   * For each *other* **candidate file** (other `survey_number`s whose bbox is near the point), do the **same single** axis-wise nearest selection (one cell per file), then compute the same degree-space Euclidean distance.
   * Take the **minimum** across those candidates (including the own file).

So yes: `.sel(..., method="nearest")` is a **single selection per file** (even though internally it finds nearest along lon and lat separately). And `closest_deg_distance` is computed by repeating that **once per candidate file**, then taking the minimum.

