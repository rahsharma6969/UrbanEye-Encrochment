import argparse, os, time, json, requests, geopandas as gpd, pandas as pd
from shapely.geometry import box
from pathlib import Path

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def load_aoi_union(aoi_dir: str) -> gpd.GeoDataFrame:
    files = list(Path(aoi_dir).glob("*.geojson"))
    if not files:
        raise SystemExit(f"No AOI files in {aoi_dir}")
    gdfs = []
    for f in files:
        g = gpd.read_file(f)
        g = g[g.geometry.notna()]
        if not g.empty:
            gdfs.append(g.to_crs("EPSG:4326"))
    merged = pd.concat(gdfs, ignore_index=True).explode(index_parts=False, ignore_index=True)
    try:
        u = merged.geometry.union_all()
    except AttributeError:
        u = merged.geometry.unary_union
    return gpd.GeoDataFrame(geometry=[u], crs="EPSG:4326")

def aoi_bbox(aoi_gdf: gpd.GeoDataFrame, expand_deg=0.001):
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    return (miny - expand_deg, minx - expand_deg, maxy + expand_deg, maxx + expand_deg)  # (S,W,N,E)

def overpass_query(bbox, q_body, max_tries=4, sleep_s=8):
    # bbox format: (S,W,N,E)
    q = f"[out:json][timeout:120];\n" + q_body.format(s=bbox[0], w=bbox[1], n=bbox[2], e=bbox[3]) + "\nout body; >; out skel qt;"
    tries = 0
    while True:
        tries += 1
        r = requests.post(OVERPASS_URL, data={"data": q}, timeout=180)
        if r.status_code == 429 or (r.status_code == 400 and "rate_limited" in r.text.lower()):
            if tries >= max_tries:
                r.raise_for_status()
            time.sleep(sleep_s * tries)
            continue
        r.raise_for_status()
        return r.json()

def osmjson_to_gdf(osm_json) -> gpd.GeoDataFrame:
    # Use geopandas' built-in helper if installed (since 0.14): gpd.read_file("OSMURL") not viable here.
    # We'll convert via osmnx-like routine using nodes/ways/relations minimally for polygons/lines.
    import networkx as nx  # only used to stitch ways quickly; if not available, fall back to simple method
    elements = osm_json.get("elements", [])
    nodes = {el["id"]: (el["lon"], el["lat"]) for el in elements if el["type"] == "node"}
    ways = [el for el in elements if el["type"] == "way"]
    rels = [el for el in elements if el["type"] == "relation"]
    # Build geometries from ways (LineString or Polygon if closed)
    from shapely.geometry import LineString, Polygon
    records = []
    for w in ways:
        nds = w.get("nodes", [])
        coords = [(nodes[n][0], nodes[n][1]) for n in nds if n in nodes]
        if len(coords) < 2:
            continue
        tags = w.get("tags", {})
        if len(coords) >= 4 and coords[0] == coords[-1] and any(k in tags for k in ["area", "landuse", "natural", "water", "leisure", "building", "amenity"]):
            geom = Polygon(coords)
        else:
            geom = LineString(coords)
        rec = {"geometry": geom}
        rec.update(tags)
        records.append(rec)
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    # Relations (multipolygons) are trickier; keep ways for now (good enough for roads/water/wetlands in most AOIs)
    return gdf

def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    aoi = load_aoi_union(a.aoi_dir)
    bbox = aoi_bbox(aoi, expand_deg=a.expand_bbox_deg)

    # Queries (union of relevant features). Overpass bbox placeholder {s},{w},{n},{e}
    q_roads = """
    (
      way["highway"]({s},{w},{n},{e});
    );
    """
    q_buildings = """
    (
      way["building"]({s},{w},{n},{e});
    );
    """
    q_wetlands = """
    (
      way["natural"="wetland"]({s},{w},{n},{e});
      way["wetland"="mangrove"]({s},{w},{n},{e});
      way["natural"="wood"]({s},{w},{n},{e})["leaf_type"="mangrove"];
      relation["natural"="wetland"]({s},{w},{n},{e});
      relation["wetland"="mangrove"]({s},{w},{n},{e});
    );
    """
    q_water = """
    (
      way["natural"="water"]({s},{w},{n},{e});
      way["waterway"]({s},{w},{n},{e});
      way["landuse"="reservoir"]({s},{w},{n},{e});
      relation["natural"="water"]({s},{w},{n},{e});
      relation["waterway"]({s},{w},{n},{e});
    );
    """
    q_protected = """
    (
      way["boundary"="protected_area"]({s},{w},{n},{e});
      relation["boundary"="protected_area"]({s},{w},{n},{e});
      way["leisure"="nature_reserve"]({s},{w},{n},{e});
      relation["leisure"="nature_reserve"]({s},{w},{n},{e});
    );
    """

    print("Fetching roads…")
    roads_json = overpass_query(bbox, q_roads)
    print("Fetching buildings…")
    bld_json = overpass_query(bbox, q_buildings)
    print("Fetching wetlands/mangroves…")
    wet_json = overpass_query(bbox, q_wetlands)
    print("Fetching water bodies…")
    water_json = overpass_query(bbox, q_water)
    print("Fetching protected areas…")
    prot_json = overpass_query(bbox, q_protected)

    roads = osmjson_to_gdf(roads_json)
    blds  = osmjson_to_gdf(bld_json)
    wets  = osmjson_to_gdf(wet_json)
    water = osmjson_to_gdf(water_json)
    prot  = osmjson_to_gdf(prot_json)

    # Clip all to AOI
    def clip(g):
        if g.empty: return g
        return g.to_crs("EPSG:4326").clip(aoi.to_crs("EPSG:4326").geometry.iloc[0])

    roads = clip(roads); blds = clip(blds); wets = clip(wets); water = clip(water); prot = clip(prot)

    # Save
    roads.to_file(os.path.join(a.out_dir, "osm_roads.geojson"), driver="GeoJSON")
    blds.to_file(os.path.join(a.out_dir, "osm_buildings.geojson"), driver="GeoJSON")
    wets.to_file(os.path.join(a.out_dir, "osm_wetlands.geojson"), driver="GeoJSON")
    water.to_file(os.path.join(a.out_dir, "osm_water.geojson"), driver="GeoJSON")
    prot.to_file(os.path.join(a.out_dir, "osm_protected.geojson"), driver="GeoJSON")

    print("Saved OSM layers to", a.out_dir)
    for name, g in [("roads",roads),("buildings",blds),("wetlands",wets),("water",water),("protected",prot)]:
        print(f"  {name}: {len(g)} features")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi_dir", default="data/aoi")
    ap.add_argument("--out_dir", default="data/context/osm")
    ap.add_argument("--expand_bbox_deg", type=float, default=0.001)  # ~100m pad
    args = ap.parse_args()
    main(args)
