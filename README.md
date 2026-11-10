# Windhoek Informal Settlement Expansion Risk

Streamlit app: logistic regression + random forest on 7 spatial predictors, with a risk map,
parcel predictor and model comparison. Part of a UNAM B.Sc Data Science (Honours) research project.

## Run locally
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    streamlit run app.py

## Build the real dataset (replaces the placeholder)
1. Get an informal-settlement POLYGON layer (GeoJSON, WGS84) and save it as `data/informal_settlements.geojson`
   (digitise in QGIS over satellite imagery, or request it from the City of Windhoek / Namibia Statistics Agency).
   If the file is missing the script tries OpenStreetMap `informal=yes` areas, which may be incomplete.
2. One-time Earth Engine login: `earthengine authenticate`
3. Run: `python build_dataset.py --project YOUR_GCP_PROJECT --t0 2018 --t1 2025 --min-new 0.05`
4. Open `data/qa_sample.csv`, check ~80 cells on satellite imagery and fill `your_verdict`.
5. Restart the app. It trains on `data/windhoek_grid.csv` automatically and the placeholder warning disappears.

## Data
Put your grid at `data/windhoek_grid.csv` with columns:
`lat, lon, RoadDist, SettleDist, PopDensity, Elevation, Slope, CBDDist, InfraProx, Expansion`.
Without it the app uses a clearly-labelled placeholder dataset.

## Google Earth Engine setup (one-off, free for research/non-commercial)
1. Register at earthengine.google.com/signup and create/select a Google Cloud project with the Earth Engine API enabled.
2. Create a service account in that project, add the role "Earth Engine Resource Viewer" (or Writer), create a JSON key, and
   register the account for Earth Engine use.
3. Local: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill it in (or run `earthengine authenticate`
   and set only `gee_project`). Streamlit Cloud: paste the same TOML into App settings > Secrets.
Without GEE the app still runs: elevation/slope fall back to OpenTopoData and PopDensity is entered manually.

## Informal settlement layer
Add `data/informal_settlements.geojson` (Points or Polygons with a `name` property). Without it, approximate built-in nodes are used.

## Deploy (free)
1. Push this folder to a GitHub repo.
2. Go to share.streamlit.io, sign in with GitHub, choose the repo, set main file to `app.py`, deploy.
