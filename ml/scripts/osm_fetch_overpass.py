

# scripts/osm_fetch_overpass.py
import argparse
import requests
import geopandas as gpd
import pandas as pd
from pathlib import Path
import time
import json

# ✅ Use kumi.systems — but only for small AOIs
OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"

def load_aoi_union(aoi_dir: str) -> gpd.GeoDataFrame:
    aoi_dir = Path(aoi_dir)
    files = list(aoi_dir.glob("*.geojson"))
    if not files:
        raise SystemExit(f"No AOI files in {aoi_dir}")
    gdfs = []
    for f in files:
        g = gpd.read_file(f)
        g = g[g.geometry.notna()]
        if not g.empty:
            gdfs.append(g.to_crs("EPSG:4326"))
    if not gdfs:
        raise SystemExit("No valid geometries in AOI files.")
    merged = pd.concat(gdfs, ignore_index=True).explode(ignore_index=True)
    try:
        u = merged.geometry.union_all()
    except AttributeError:
        u = merged.geometry.unary_union
    if not u.is_valid:
        u = u.buffer(0)  # Fix invalid geometry
    return gpd.GeoDataFrame(geometry=[u], crs="EPSG:4326")

def aoi_bbox(aoi_gdf: gpd.GeoDataFrame):
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    return miny, minx, maxy, maxx  # S, W, N, E

def overpass_query(bbox, query_body, timeout=60):
    S, W, N, E = bbox
    query = f"""
    [out:json][timeout:{timeout}];
    {query_body.format(s=S, w=W, n=N, e=E)}
    out body; >; out skel qt;
    """.strip()

    print(f"📤 Sending query to {OVERPASS_URL}")
    print(f"🔍 Query: {query[:200]}...")

    for i in range(3):
        try:
            r = requests.post(OVERPASS_URL, data={"data": query}, timeout=timeout + 10)
            if r.status_code == 200:
                try:
                    data = r.json()
                    if "elements" in data:
                        print(f"✅ Success! Got {len(data['elements'])} elements")
                        return data
                    else:
                        print("❌ No 'elements' in response.")
                        return {"elements": []}
                except json.JSONDecodeError:
                    print("❌ Response is not JSON:", r.text[:200])
                    return {"elements": []}
            else:
                print(f"❌ HTTP {r.status_code}: {r.text[:200]}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Request failed (attempt {i+1}/3): {e}")
            if i == 2:
                print("❌ Giving up after 3 attempts.")
                return {"elements": []}
            time.sleep(2)
    return {"elements": []}

def osmjson_to_gdf(osm_json) -> gpd.GeoDataFrame:
    elements = osm_json.get("elements", [])
    nodes = {el["id"]: (el["lon"], el["lat"]) for el in elements if el["type"] == "node"}
    ways = [el for el in elements if el["type"] == "way"]
    records = []
    for w in ways:
        nds = w.get("nodes", [])
        coords = [(nodes[n][0], nodes[n][1]) for n in nds if n in nodes]
        if len(coords) < 2:
            continue
        tags = w.get("tags", {})
        if len(coords) >= 4 and coords[0] == coords[-1]:
            try:
                from shapely import Polygon
                geom = Polygon(coords)
            except:
                continue
        else:
            try:
                from shapely import LineString
                geom = LineString(coords)
            except:
                continue
        records.append({"geometry": geom, **tags})
    return gpd.GeoDataFrame(records, crs="EPSG:4326")

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aoi = load_aoi_union(args.aoi_dir)
    bbox = aoi_bbox(aoi)
    print(f"📍 AOI BBox (S,W,N,E): {bbox}")

    q_buildings = """
    (
      way["building"]({s},{w},{n},{e});
    );
    """
    q_roads = """
    (
      way["highway"]({s},{w},{n},{e});
    );
    """

    print("🔍 Fetching buildings...")
    bld_json = overpass_query(bbox, q_buildings)
    print("🔍 Fetching roads...")
    road_json = overpass_query(bbox, q_roads)

    buildings = osmjson_to_gdf(bld_json)
    roads = osmjson_to_gdf(road_json)

    def safe_clip(gdf):
        if gdf.empty:
            return gdf
        gdf = gdf.copy()
        gdf['geometry'] = gdf.buffer(0)  # Fix invalid
        return gdf.clip(aoi)

    buildings = safe_clip(buildings)
    roads = safe_clip(roads)

    buildings.to_file(out_dir / "osm_buildings.geojson", driver="GeoJSON")
    roads.to_file(out_dir / "osm_roads.geojson", driver="GeoJSON")

    print(f"✅ Saved to {out_dir}")
    print(f"  Buildings: {len(buildings)}")
    print(f"  Roads: {len(roads)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi_dir", default="data/aoi")
    parser.add_argument("--out_dir", default="data/context/osm")
    args = parser.parse_args()
    main(args)
    
    
'''
    python scripts/osm_fetch_overpass.py ^
  --aoi_dir data/aoi ^
  --out_dir data/context/osm
'''