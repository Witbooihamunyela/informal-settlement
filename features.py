"""Extract the 7 thesis predictors for ANY clicked location in Windhoek.

Sources (all free):
  Elevation, Slope  -> Google Earth Engine  USGS/SRTMGL1_003 (30 m)   [fallback: OpenTopoData SRTM]
  PopDensity        -> Google Earth Engine  WorldPop/GP/100m/pop (NAM, 2020), 500 m radius
  RoadDist          -> OpenStreetMap (Overpass): motorway/trunk/primary/secondary roads
  InfraProx         -> OpenStreetMap (Overpass): schools, clinics, hospitals, police, water points
  SettleDist        -> data/informal_settlements.geojson (your digitised layer) or approximate nodes
  CBDDist           -> great-circle distance to the CBD coordinate below
This module does not import streamlit, so it can be unit-tested on its own.
"""
import json
import math
import os

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
NODES_PATH = os.path.join(HERE, "data", "informal_settlements.geojson")
CBD = (-22.5650, 17.0830)  # lat, lon - same reference point as the thesis map
# APPROXIMATE centroids. Replace with a digitised layer (QGIS / City of Windhoek) for real work.
DEFAULT_NODES = {"Havana": (-22.500, 17.030), "Goreangab": (-22.540, 17.030),
                 "Okuryangava": (-22.470, 17.050), "Otjomuise ext.": (-22.580, 17.030)}
OVERPASS = "https://overpass-api.de/api/interpreter"
UA = {"User-Agent": "windhoek-expansion-app/0.1 (student research)"}


# ---------- geometry ----------
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


def _xy(lat, lon, lat0):
    return lon * 111.320 * math.cos(math.radians(lat0)), lat * 110.574


def _pt_seg(p, a, b):
    ax, ay, bx, by, px, py = *a, *b, *p
    dx, dy = bx - ax, by - ay
    t = 0 if dx == dy == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def dist_to_polylines_km(lat, lon, lines):
    p = _xy(lat, lon, lat)
    best = float("inf")
    for line in lines:
        pts = [_xy(a, b, lat) for a, b in line]
        for i in range(len(pts) - 1):
            best = min(best, _pt_seg(p, pts[i], pts[i + 1]))
    return best


# ---------- settlements / CBD ----------
def load_nodes():
    if os.path.exists(NODES_PATH):
        gj = json.load(open(NODES_PATH))
        out = {}
        for i, f in enumerate(gj["features"]):
            g = f["geometry"]
            c = g["coordinates"] if g["type"] == "Point" else g["coordinates"][0][0] if g["type"] == "Polygon" else None
            if c is not None:
                out[f.get("properties", {}).get("name", f"settlement {i}")] = (c[1], c[0])
        return out, "your informal_settlements.geojson"
    return DEFAULT_NODES, "APPROXIMATE built-in nodes (add data/informal_settlements.geojson)"


def settle_dist(lat, lon):
    nodes, src = load_nodes()
    name, d = min(((n, haversine_km(lat, lon, *c)) for n, c in nodes.items()), key=lambda x: x[1])
    return d, f"nearest: {name}; source: {src}"


# ---------- OpenStreetMap ----------
MIRRORS = [OVERPASS, "https://overpass.kumi.systems/api/interpreter"]


def _overpass(q):
    last = None
    for url in MIRRORS:
        try:
            r = requests.post(url, data={"data": q}, timeout=20, headers=UA)
            r.raise_for_status()
            return r.json()["elements"]
        except Exception as e:
            last = e
    raise last


def osm_road_dist(lat, lon, radius=8000):
    q = (f'[out:json][timeout:15];way["highway"~"^(motorway|trunk|primary|secondary)$"]'
         f'(around:{radius},{lat},{lon});out geom;')
    lines = [[(g["lat"], g["lon"]) for g in e["geometry"]] for e in _overpass(q) if "geometry" in e]
    return (dist_to_polylines_km(lat, lon, lines), f"OSM major roads, {len(lines)} segments in {radius/1000:.0f} km") \
        if lines else (None, "no major roads found within search radius")


def osm_infra_dist(lat, lon, radius=10000):
    q = (f'[out:json][timeout:15];(nwr["amenity"~"^(school|clinic|hospital|police)$"](around:{radius},{lat},{lon});'
         f'nwr["man_made"~"^(water_tap|water_well|water_tower)$"](around:{radius},{lat},{lon}););out center;')
    pts = []
    for e in _overpass(q):
        la, lo = e.get("lat"), e.get("lon")
        if la is None and "center" in e:
            la, lo = e["center"]["lat"], e["center"]["lon"]
        if la is not None:
            pts.append((la, lo))
    return (min(haversine_km(lat, lon, a, b) for a, b in pts), f"OSM schools/clinics/police/water, {len(pts)} found") \
        if pts else (None, "no infrastructure found within search radius")


# ---------- Google Earth Engine ----------
def init_ee(service_account=None, project=None):
    import ee
    if service_account:
        creds = ee.ServiceAccountCredentials(service_account["client_email"], key_data=json.dumps(dict(service_account)))
        ee.Initialize(creds, project=service_account.get("project_id") or project)
    elif project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()
    return ee


def gee_terrain_pop(ee, lat, lon):
    pt = ee.Geometry.Point([lon, lat])
    srtm = ee.Image("USGS/SRTMGL1_003")
    v = srtm.addBands(ee.Terrain.slope(srtm)).reduceRegion(ee.Reducer.mean(), pt.buffer(45), 30).getInfo()
    pop = (ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq("country", "NAM"))
           .filter(ee.Filter.eq("year", 2020)).mosaic().select("population"))
    s = pop.reduceRegion(ee.Reducer.sum(), pt.buffer(500), 100).get("population").getInfo()
    return v.get("elevation"), v.get("slope"), (None if s is None else s / (math.pi * 0.5 ** 2))


def opentopo_terrain(lat, lon, step_m=30):
    dlat, dlon = step_m / 110574.0, step_m / (111320.0 * math.cos(math.radians(lat)))
    pts = [(lat, lon), (lat + dlat, lon), (lat - dlat, lon), (lat, lon + dlon), (lat, lon - dlon)]
    r = requests.get("https://api.opentopodata.org/v1/srtm30m",
                     params={"locations": "|".join(f"{a},{b}" for a, b in pts)}, timeout=30, headers=UA)
    z = [x["elevation"] for x in r.json()["results"]]
    dzdy, dzdx = (z[1] - z[2]) / (2 * step_m), (z[3] - z[4]) / (2 * step_m)
    return z[0], math.degrees(math.atan(math.hypot(dzdx, dzdy)))


def ee_tile_urls(ee):
    srtm = ee.Image("USGS/SRTMGL1_003")
    aoi = ee.Geometry.Rectangle([16.95, -22.70, 17.25, -22.40])
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi)
          .filterDate("2025-01-01", "2026-09-30").filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10)).median())
    pop = ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq("country", "NAM")).filter(ee.Filter.eq("year", 2020)).mosaic()
    layers = {
        "GEE Sentinel-2 true colour": (s2, {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}),
        "GEE SRTM slope (deg)": (ee.Terrain.slope(srtm), {"min": 0, "max": 35, "palette": ["white", "yellow", "orange", "red"]}),
        "GEE WorldPop density": (pop, {"min": 0, "max": 50, "palette": ["white", "yellow", "orange", "red", "purple"]}),
    }
    return {k: img.getMapId(vis)["tile_fetcher"].url_format for k, (img, vis) in layers.items()}


# ---------- everything for one click ----------
def _terrain(ee, lat, lon):
    vals, notes = {"Elevation": None, "Slope": None, "PopDensity": None}, {}
    if ee is not None:
        try:
            vals["Elevation"], vals["Slope"], vals["PopDensity"] = gee_terrain_pop(ee, lat, lon)
            notes["Elevation"] = notes["Slope"] = "Google Earth Engine SRTM 30 m"
            notes["PopDensity"] = "Google Earth Engine WorldPop 100 m (2020), 500 m radius"
        except Exception as e:
            notes["Elevation"] = f"GEE failed ({type(e).__name__})"
    if vals["Elevation"] is None:
        try:
            vals["Elevation"], vals["Slope"] = opentopo_terrain(lat, lon)
            notes["Elevation"] = notes["Slope"] = "OpenTopoData SRTM 30 m (fallback)"
        except Exception as e:
            notes["Elevation"] = notes["Slope"] = f"lookup failed ({type(e).__name__}) - enter manually"
    notes.setdefault("PopDensity", "needs Google Earth Engine - enter manually")
    return vals, notes


def extract_all(lat, lon, ee=None):
    from concurrent.futures import ThreadPoolExecutor
    vals, notes = {}, {}
    vals["CBDDist"], notes["CBDDist"] = haversine_km(lat, lon, *CBD), "distance to CBD reference point"
    vals["SettleDist"], notes["SettleDist"] = settle_dist(lat, lon)
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_road = ex.submit(osm_road_dist, lat, lon)
        f_infra = ex.submit(osm_infra_dist, lat, lon)
        f_terr = ex.submit(_terrain, ee, lat, lon)
        for key, fut in (("RoadDist", f_road), ("InfraProx", f_infra)):
            try:
                vals[key], notes[key] = fut.result()
            except Exception as e:
                vals[key], notes[key] = None, f"lookup failed ({type(e).__name__}) - enter manually"
        tv, tn = f_terr.result()
    vals.update(tv)
    notes.update(tn)
    return vals, notes


# ---------- place name (OpenStreetMap Nominatim; max ~1 request/second, please keep caching on) ----------
def reverse_geocode(lat, lon):
    """Return a readable place name such as 'Okuryangava, Katutura, Windhoek', or None."""
    try:
        r = requests.get("https://nominatim.openstreetmap.org/reverse", timeout=15, headers=UA,
                         params={"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 16, "addressdetails": 1})
        j = r.json()
        a = j.get("address", {})
        parts = []
        for k in ("road", "neighbourhood", "suburb", "city_district", "city", "town", "village"):
            v = a.get(k)
            if v and v not in parts:
                parts.append(v)
        return ", ".join(parts[:3]) or j.get("display_name")
    except Exception:
        return None
