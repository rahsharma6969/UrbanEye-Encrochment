# scripts/check_chip_context.py
import os, argparse
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.warp import transform_bounds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip_id", required=True)
    ap.add_argument("--parquet", default="outputs/chips_index_fast.parquet")
    ap.add_argument("--osm_buildings", default="data/context/osm/osm_buildings.geojson")
    ap.add_argument("--osm_roads", default="data/context/osm/osm_roads.geojson")
    ap.add_argument("--worldcover", default="data/context/worldcover.tif")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.parquet)
    row = df[df.chip_id==args.chip_id].iloc[0]
    xmin,xmin,ymin,xmax,ymax = row.xmin, row.ymin, row.xmax, row.ymax
    crs = row.crs
    chip_geom = box(xmin, ymin, xmax, ymax)

    print("Chip CRS:", crs, "bounds:", chip_geom.bounds)

    if os.path.exists(args.osm_buildings):
        b = gpd.read_file(args.osm_buildings)
        if b.crs is None: b.set_crs("EPSG:4326", inplace=True)
        b2 = b.to_crs(crs)
        print("Buildings intersecting:", len(b2[b2.intersects(chip_geom)]))
    else:
        print("No osm buildings file")

    if os.path.exists(args.osm_roads):
        r = gpd.read_file(args.osm_roads)
        if r.crs is None: r.set_crs("EPSG:4326", inplace=True)
        r2 = r.to_crs(crs)
        print("Road features intersecting:", len(r2[r2.intersects(chip_geom)]))
    else:
        print("No osm roads file")

    if os.path.exists(args.worldcover):
        with rasterio.open(args.worldcover) as src:
            wc_crs = src.crs
            print("Worldcover CRS:", wc_crs, "dtype:", src.dtypes[0])
            # get a tiny sample by reading whole file is heavy — so just convert chip bounds to wc_crs for info
            b_trans = transform_bounds(crs, wc_crs, xmin, ymin, xmax, ymax)
            print("Chip bounds in worldcover CRS:", b_trans)
    else:
        print("No worldcover file")

if __name__ == "__main__":
    main()


'''
python scripts\check_chips_context.py --chip_id s2_0_50 --parquet outputs\mumbai_index.parquet
'''