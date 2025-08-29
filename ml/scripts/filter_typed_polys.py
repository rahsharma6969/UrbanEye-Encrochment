import argparse, geopandas as gpd, os
def main(a):
    gdf = gpd.read_file(a.in_geojson)
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    gdf["area_m2"] = gdf.area
    gdf = gdf.to_crs(4326)
    q = gdf[gdf["area_m2"] >= a.min_area_m2].copy()
    os.makedirs(a.out_dir, exist_ok=True)
    out_geo = os.path.join(a.out_dir, "alerts_typed.geojson")
    q.to_file(out_geo, driver="GeoJSON")
    print(f"Typed alerts (≥{a.min_area_m2} m²): {len(q)} → {out_geo}")
    print(q["change_type"].value_counts())
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_geojson", default="outputs/polys_typed/changes_typed.geojson")
    ap.add_argument("--min_area_m2", type=float, default=200)
    ap.add_argument("--out_dir", default="outputs/alerts_typed")
    args = ap.parse_args(); main(args)