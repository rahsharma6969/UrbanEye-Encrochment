import argparse, os, numpy as np, geopandas as gpd, pandas as pd, rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box

# WorldCover class names (v2021/2023)
WC_NAMES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse",
    70: "Snow / ice",
    80: "Water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100:"Moss / lichen",
}

VEG = {10,20,30}
WET = {90,95}
WATER = {80}
BUILT = {50}
AGRI = {40}
BARE = {60}

def zonal_hist_wcs(poly_wgs, wc_path):
    with rasterio.open(wc_path) as src:
        poly_src = gpd.GeoSeries([poly_wgs], crs="EPSG:4326").to_crs(src.crs).iloc[0]
        try:
            arr, _ = rio_mask(src, [poly_src.__geo_interface__], crop=True, nodata=0, filled=True)
        except Exception:
            return {}
        a = arr[0].astype("int32")
        a = a[(a!=0) & np.isfinite(a)]
        if a.size == 0:
            return {}
        vals, counts = np.unique(a, return_counts=True)
        return {int(v): int(c) for v,c in zip(vals, counts)}

def frac(d, cls_set):
    total = sum(d.values()) if d else 0
    if total == 0: return 0.0
    return float(sum(c for k,c in d.items() if k in cls_set)) / float(total)

def choose_change_type(h):
    if not h: return "unknown"
    total = sum(h.values())
    if total == 0: return "unknown"

    f_built = frac(h, BUILT)
    f_water = frac(h, WATER)
    f_wet   = frac(h, WET)
    f_veg   = frac(h, VEG)
    f_agri  = frac(h, AGRI)
    f_bare  = frac(h, BARE)

    # simple rule set (tweakable)
    if f_built >= 0.30:
        return "built-up change"
    if (f_water + f_wet) >= 0.30:
        return "water/wetland change"
    if f_veg >= 0.40:
        return "vegetation change"
    if f_agri >= 0.40:
        return "agriculture change"
    if f_bare >= 0.40:
        return "bare/soil change"

    # fallback to dominant class
    dom = max(h.items(), key=lambda kv: kv[1])[0]
    return f"{WC_NAMES.get(dom, 'unknown')} change"

def main(a):
    os.makedirs(a.out_dir, exist_ok=True)

    alerts = gpd.read_file(a.alerts_geojson).to_crs("EPSG:4326")
    if alerts.empty:
        print("No alerts found in input.")
        return
    print(f"Loaded {len(alerts)} alerts")

    # Optional overlays
    bld = None
    if a.ms_buildings and os.path.exists(a.ms_buildings):
        bld = gpd.read_file(a.ms_buildings)
        bld = bld.to_crs("EPSG:4326")
        print(f"Loaded buildings: {len(bld)}")

    roads = None
    if a.osm_roads and os.path.exists(a.osm_roads):
        roads = gpd.read_file(a.osm_roads).to_crs("EPSG:4326")
        if a.road_buffer_m > 0:
            roads_m = roads.to_crs(roads.estimate_utm_crs())
            roads_buf_m = roads_m.buffer(a.road_buffer_m)
            roads = gpd.GeoDataFrame(geometry=roads_buf_m, crs=roads_m.crs).to_crs("EPSG:4326")
        print(f"Loaded roads: {len(roads)} (buffered {a.road_buffer_m} m)")

    # Tag each alert
    wc_hist_list = []
    dom_class_list = []
    dom_name_list = []
    built_frac_list = []
    veg_frac_list = []
    wet_frac_list = []
    water_frac_list = []
    agri_frac_list = []
    change_type_list = []
    overlaps_building = []
    overlaps_road = []

    for geom in alerts.geometry:
        # WorldCover histogram within polygon
        h = zonal_hist_wcs(geom, a.worldcover_tif)
        wc_hist_list.append(h)

        if h:
            dom = max(h.items(), key=lambda kv: kv[1])[0]
            dom_class_list.append(int(dom))
            dom_name_list.append(WC_NAMES.get(int(dom), "Unknown"))
        else:
            dom_class_list.append(None)
            dom_name_list.append("Unknown")

        built_frac_list.append(frac(h, BUILT))
        veg_frac_list.append(frac(h, VEG))
        wet_frac_list.append(frac(h, WET))
        water_frac_list.append(frac(h, WATER))
        agri_frac_list.append(frac(h, AGRI))

        ctype = choose_change_type(h)
        change_type_list.append(ctype)

        # overlaps with buildings / roads (optional)
        if bld is not None and not bld.empty:
            overlaps_building.append(bld.intersects(geom).any())
        else:
            overlaps_building.append(False)

        if roads is not None and not roads.empty:
            overlaps_road.append(roads.intersects(geom).any())
        else:
            overlaps_road.append(False)

    alerts["wc_hist"] = [str(h) for h in wc_hist_list]
    alerts["wc_dom_class"] = dom_class_list
    alerts["wc_dom_name"] = dom_name_list
    alerts["wc_frac_built"] = built_frac_list
    alerts["wc_frac_veg"] = veg_frac_list
    alerts["wc_frac_wet"] = wet_frac_list
    alerts["wc_frac_water"] = water_frac_list
    alerts["wc_frac_agri"] = agri_frac_list
    alerts["overlaps_building"] = overlaps_building
    alerts["overlaps_road"] = overlaps_road

    # refine change_type with overlays
    refined = []
    for i, ct in enumerate(change_type_list):
        if overlaps_building[i] or alerts.loc[i, "wc_frac_built"] >= 0.30:
            refined.append("building/impervious change")
        elif overlaps_road[i]:
            refined.append("road/linear development")
        else:
            refined.append(ct)
    alerts["change_type"] = refined

    # save enriched outputs
    out_geo = os.path.join(a.out_dir, "alerts_typed.geojson")
    out_csv = os.path.join(a.out_dir, "alerts_typed.csv")
    alerts.to_file(out_geo, driver="GeoJSON")
    alerts.drop(columns="geometry").to_csv(out_csv, index=False)

    print(f"Tagged {len(alerts)} alerts → {out_geo}")
    print("Top change types:\n", alerts["change_type"].value_counts().head(10))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts_geojson", default="outputs/alerts_loose/alerts.geojson")
    ap.add_argument("--worldcover_tif", default="data/context/worldcover.tif")  # required
    ap.add_argument("--ms_buildings", default="")  # optional GeoJSON/GeoPackage
    ap.add_argument("--osm_roads", default="")     # optional
    ap.add_argument("--road_buffer_m", type=float, default=6.0)
    ap.add_argument("--out_dir", default="outputs/alerts_typed")
    args = ap.parse_args()
    main(args)
