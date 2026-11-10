"""Build the REAL Windhoek grid dataset from Google Earth Engine + OpenStreetMap -> data/windhoek_grid.csv

Two ways to run it:
  * inside the Streamlit app (button "Build dataset and train"), or
  * on the command line:  python build_dataset.py --project YOUR_GCP_PROJECT --t0 2018 --t1 2025

Design:
  Grid        50 x 50 cells over lat -22.65..-22.45, lon 17.00..17.18 (same as the thesis).
  Label Y     1 if >= min_new of the cell's area became NEW built-up between t0 and t1
              (Google Dynamic World 'built' class, 10 m, yearly mode composite).
              If an informal-settlement polygon layer is available (data/informal_settlements.geojson, or OSM
              `informal=yes` areas) only new built-up INSIDE that footprint counts. If NOT, the label is
              all new built-up growth (urban expansion) - the metadata file records which definition was used.
  Predictors  all measured at t0 so the label cannot leak into them:
              Elevation, Slope   SRTM 30 m            | PopDensity  WorldPop 100 m, year t0
              SettleDist         distance to (informal) built-up that already existed at t0
              RoadDist, InfraProx  OpenStreetMap (today's snapshot)   | CBDDist  distance to CBD reference
Known limits: Dynamic World can confuse bare ground with built-up in arid land, so check data/qa_sample.csv on
satellite imagery. Without a settlement layer the label is built-up growth, not specifically informal growth.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
INFORMAL_PATH = os.path.join(DATA, "informal_settlements.geojson")
LAT0, LAT1, LON0, LON1 = -22.65, -22.45, 17.00, 17.18
N_SIDE = 50
CBD = (-22.5650, 17.0830)
PAD = 0.03
FEATURES = ["RoadDist", "SettleDist", "PopDensity", "Elevation", "Slope", "CBDDist", "InfraProx"]
UA = {"User-Agent": "windhoek-expansion-app/0.3 (student research)"}
MIRRORS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
TABLE_4_1 = {"RoadDist": 2.839, "SettleDist": 8.086, "PopDensity": 1317.0, "Elevation": 1769.6,
             "Slope": 25.70, "CBDDist": 8.074, "InfraProx": 5.999}


# ---------------- grid ----------------
def make_grid(n=N_SIDE):
    dlat, dlon = (LAT1 - LAT0) / n, (LON1 - LON0) / n
    rows = []
    for i in range(n):
        for j in range(n):
            s, w = LAT0 + i * dlat, LON0 + j * dlon
            rows.append(dict(row=i, col=j, lat=s + dlat / 2, lon=w + dlon / 2, s=s, n=s + dlat, w=w, e=w + dlon))
    return pd.DataFrame(rows)


# ---------------- geometry helpers ----------------
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def _xy(lat, lon, lat0=-22.55):
    return np.column_stack([np.asarray(lon) * 111.320 * math.cos(math.radians(lat0)), np.asarray(lat) * 110.574])


def min_dist_to_segments(P, A, B, chunk=100):
    """Distance (km) from each point in P (n,2) to the nearest segment A[k]-B[k] (all in km)."""
    out = np.empty(len(P))
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk, None, :]
        t = np.clip(((p - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
        proj = A[None] + t[..., None] * AB[None]
        out[s:s + chunk] = np.sqrt(((p - proj) ** 2).sum(2)).min(1)
    return out


def make_label(new_frac, min_new):
    return (np.asarray(new_frac) >= min_new).astype(int)


# ---------------- OpenStreetMap ----------------
def overpass(q):
    last = None
    for url in MIRRORS:
        try:
            r = requests.post(url, data={"data": q}, timeout=180, headers=UA)
            r.raise_for_status()
            return r.json()["elements"]
        except Exception as e:
            last = e
    raise RuntimeError(f"Overpass failed: {last}")


def _bbox():
    return f"{LAT0 - PAD},{LON0 - PAD},{LAT1 + PAD},{LON1 + PAD}"


def osm_road_dist(grid, log=print):
    els = overpass(f'[out:json][timeout:120];way["highway"~"^(motorway|trunk|primary|secondary)$"]({_bbox()});out geom;')
    A, B = [], []
    for e in els:
        g = e.get("geometry", [])
        for k in range(len(g) - 1):
            A.append((g[k]["lat"], g[k]["lon"]))
            B.append((g[k + 1]["lat"], g[k + 1]["lon"]))
    if not A:
        raise RuntimeError("No major roads returned by OpenStreetMap")
    A, B = np.array(A), np.array(B)
    log(f"  roads: {len(els)} ways, {len(A)} segments")
    return min_dist_to_segments(_xy(grid.lat, grid.lon), _xy(A[:, 0], A[:, 1]), _xy(B[:, 0], B[:, 1]))


def osm_infra_dist(grid, log=print):
    bb = _bbox()
    els = overpass(f'[out:json][timeout:120];(nwr["amenity"~"^(school|clinic|hospital|police)$"]({bb});'
                   f'nwr["man_made"~"^(water_tap|water_well|water_tower)$"]({bb}););out center;')
    pts = []
    for e in els:
        la, lo = e.get("lat"), e.get("lon")
        if la is None and "center" in e:
            la, lo = e["center"]["lat"], e["center"]["lon"]
        if la is not None:
            pts.append((la, lo))
    if not pts:
        raise RuntimeError("No infrastructure returned by OpenStreetMap")
    pts = np.array(pts)
    log(f"  infrastructure points: {len(pts)}")
    d = haversine_km(grid.lat.values[:, None], grid.lon.values[:, None], pts[None, :, 0], pts[None, :, 1])
    return d.min(1)


def osm_informal_polygons():
    els = overpass(f'[out:json][timeout:120];(way["informal"="yes"]({_bbox()}););out geom;')
    polys = []
    for e in els:
        g = e.get("geometry", [])
        if len(g) >= 4 and g[0] == g[-1]:
            polys.append({"type": "Polygon", "coordinates": [[[p["lon"], p["lat"]] for p in g]]})
    return polys


def load_informal(path=INFORMAL_PATH, log=print):
    """Return GeoJSON polygon geometries of the informal footprint, or [] if none can be found (never raises)."""
    try:
        if os.path.exists(path):
            gj = json.load(open(path))
            polys = [f["geometry"] for f in gj["features"] if f["geometry"]["type"] in ("Polygon", "MultiPolygon")]
            log(f"  informal layer: {len(polys)} polygons from {path}")
            return polys
        polys = osm_informal_polygons()
        log(f"  no {os.path.basename(path)}; OpenStreetMap informal=yes areas found: {len(polys)}")
        return polys
    except Exception as e:
        log(f"  informal layer lookup failed ({type(e).__name__}): {e}")
        return []


# ---------------- Google Earth Engine ----------------
def ee_extract(grid, polys, t0, t1, ee, log=print):
    """Needs an INITIALISED earthengine-api module `ee`. Works in batches of 10 grid rows."""
    aoi = ee.Geometry.Rectangle([LON0 - 0.05, LAT0 - 0.05, LON1 + 0.05, LAT1 + 0.05])
    if polys:
        informal = ee.Image(0).byte().paint(ee.FeatureCollection([ee.Feature(ee.Geometry(g)) for g in polys]), 1)
    else:
        informal = ee.Image(1).byte()  # no settlement layer: every built-up pixel counts
    informal = informal.rename("informal")

    def built(y):
        dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(aoi)
              .filterDate(f"{y}-01-01", f"{y + 1}-01-01").select("label"))
        return dw.mode().eq(6).unmask(0).rename(f"built{y}")  # class 6 = built

    b0, b1 = built(t0), built(t1)
    new_informal = b1.And(b0.Not()).And(informal).rename("new_informal_frac")
    srtm = ee.Image("USGS/SRTMGL1_003")
    terrain = srtm.rename("Elevation").addBands(ee.Terrain.slope(srtm).rename("Slope"))
    pop = (ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq("country", "NAM"))
           .filter(ee.Filter.eq("year", t0)).mosaic().select("population").multiply(100).rename("PopDensity"))
    src = b0.And(informal)  # (informal) built-up that already existed at t0
    src100 = src.reduceResolution(ee.Reducer.max(), True, 1024).reproject("EPSG:32733", None, 100)
    dist = (src100.Not().fastDistanceTransform(256, "pixels", "squared_euclidean").sqrt()
            .multiply(100).divide(1000).rename("SettleDist"))  # km
    fractions = b0.rename("built_t0_frac").addBands(new_informal).addBands(informal.rename("informal_frac"))

    mean = ee.Reducer.mean()
    rows = sorted(grid.row.unique())
    batches = []
    for k in range(0, len(rows), 10):
        sub = grid[grid.row.isin(rows[k:k + 10])]
        cells = ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle([r.w, r.s, r.e, r.n]),
                                                 {"row": int(r.row), "col": int(r.col)}) for r in sub.itertuples()])
        parts = []
        for img, scale in ((terrain, 30), (pop.addBands(dist), 100), (fractions, 10)):
            feats = img.reduceRegions(cells, mean, scale=scale, tileScale=8).getInfo()["features"]
            parts.append(pd.DataFrame([f["properties"] for f in feats]))
        out = parts[0]
        for p in parts[1:]:
            out = out.merge(p, on=["row", "col"])
        batches.append(out)
        log(f"  Earth Engine: grid rows {rows[k]}-{rows[min(k + 9, len(rows) - 1)]} done ({len(batches)}/{math.ceil(len(rows) / 10)})")
    return pd.concat(batches, ignore_index=True)


# ---------------- build + save ----------------
def build(ee, t0=2018, t1=2025, min_new=0.05, polys=None, log=print):
    grid = make_grid()
    log("Step 1/3: OpenStreetMap predictors (roads, infrastructure) ...")
    df = grid[["row", "col", "lat", "lon"]].copy()
    df["RoadDist"] = osm_road_dist(grid, log)
    df["InfraProx"] = osm_infra_dist(grid, log)
    df["CBDDist"] = haversine_km(df.lat.values, df.lon.values, *CBD)
    log("Step 2/3: Google Earth Engine (terrain, population, built-up change). This takes several minutes ...")
    df = df.merge(ee_extract(grid, polys, t0, t1, ee, log), on=["row", "col"])
    df["Expansion"] = make_label(df.new_informal_frac, min_new)
    log("Step 3/3: checks ...")
    cols = ["lat", "lon"] + FEATURES + ["Expansion", "new_informal_frac", "built_t0_frac", "informal_frac"]
    meta = {"t0": t0, "t1": t1, "min_new": min_new, "n_informal_polygons": len(polys or []),
            "label_definition": ("new built-up inside the informal-settlement footprint" if polys
                                 else "ALL new built-up growth (no informal-settlement layer was available)"),
            "built_up_source": "GOOGLE/DYNAMICWORLD/V1 label==6 (yearly mode)", "terrain": "USGS/SRTMGL1_003",
            "population": "WorldPop/GP/100m/pop (NAM, year t0)", "osm": "Overpass snapshot at run time"}
    return df[cols], meta


def save(df, meta, seed=42, log=print):
    os.makedirs(DATA, exist_ok=True)
    if df.Expansion.nunique() < 2:
        raise RuntimeError("The label has only one class. Lower --min-new or check the years / settlement layer.")
    df.to_csv(os.path.join(DATA, "windhoek_grid.csv"), index=False)
    qa = pd.concat([df[df.Expansion == 1].sample(min(40, int(df.Expansion.sum())), random_state=seed),
                    df[df.Expansion == 0].sample(min(40, int((df.Expansion == 0).sum())), random_state=seed)])
    qa["check_on_map"] = [f"https://www.google.com/maps/@{la:.5f},{lo:.5f},17z/data=!3m1!1e3" for la, lo in zip(qa.lat, qa.lon)]
    qa["your_verdict"] = ""
    qa.to_csv(os.path.join(DATA, "qa_sample.csv"), index=False)
    json.dump(meta, open(os.path.join(DATA, "dataset_metadata.json"), "w"), indent=2)
    log(f"Saved data/windhoek_grid.csv: {len(df)} cells, {int(df.Expansion.sum())} expansion ({df.Expansion.mean():.1%})")
    for thr in (0.01, 0.02, 0.05, 0.10, 0.20):
        log(f"  label sensitivity: min-new {thr:.2f} -> {(df.new_informal_frac >= thr).mean():.1%} of cells")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("GEE_PROJECT"), help="Google Cloud project with Earth Engine enabled")
    ap.add_argument("--t0", type=int, default=2018)
    ap.add_argument("--t1", type=int, default=2025)
    ap.add_argument("--min-new", type=float, default=0.05)
    ap.add_argument("--informal", default=INFORMAL_PATH)
    a = ap.parse_args()
    import ee
    ee.Initialize(project=a.project) if a.project else ee.Initialize()
    polys = load_informal(a.informal)
    df, meta = build(ee, a.t0, a.t1, a.min_new, polys)
    save(df, meta)
    print(f"\n{'Feature':<11}{'mean':>10}{'(thesis T4.1)':>15}")
    for f in FEATURES:
        print(f"{f:<11}{df[f].mean():>10.2f}{TABLE_4_1[f]:>15.2f}")
    print("\nLabel definition used:", meta["label_definition"])


if __name__ == "__main__":
    main()
