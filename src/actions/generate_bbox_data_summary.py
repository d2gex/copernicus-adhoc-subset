from src.config import INPUT_ROOT, OUTPUT_ROOT
from src.utils import join_uri
from src.build_bounding_box import DatasetDigest, BoundingBoxBuilder


if __name__ == "__main__":
    INPUT = join_uri(
        INPUT_ROOT, "All_points_cetacea_megafauna_generated_points_5km.csv"
    )
    OUT_POINTS = join_uri(OUTPUT_ROOT, "points_with_index.csv")
    OUT_SUMMARY = join_uri(OUTPUT_ROOT, "date_bbox_summary.csv")

    digest = DatasetDigest(
        input_path=INPUT,
        output_points_path=OUT_POINTS,
        output_summary_path=OUT_SUMMARY,
        bbox_builder=BoundingBoxBuilder(padding_deg=0.08),
    )

    points_df, summary_df = digest.run()
    print(f"Wrote: {OUT_POINTS} ({len(points_df)} rows)")
    print(f"Wrote: {OUT_SUMMARY} ({len(summary_df)} rows)")
