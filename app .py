"""Windhoek Informal Settlement Expansion Risk - Streamlit app with an interactive map."""
import os
from datetime import date, timedelta

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium

import build_dataset as bd
import features as ft
from data_prep import CSV_PATH, FEATURES, TARGET

try:
    from streamlit_geolocation import streamlit_geolocation
except Exception:  # package not installed: the app still works, without the GPS button
    streamlit_geolocation = None

LAT0, LAT1, LON0, LON1 = -22.65, -22.45, 17.00, 17.18  # study grid bounds
MAX_BUILT = 0.5  # land that is already >= 50% built-up cannot expand any further


def _built_year():
    try:
        import json as _j
        return int(_j.load(open(os.path.join(os.path.dirname(CSV_PATH), "dataset_metadata.json")))["t1"])
    except Exception:
        return date.today().year - 1

st.set_page_config(page_title="Informal Settlement Expansion Prediction - Windhoek", page_icon="🏘️", layout="wide")
UNITS = {"RoadDist": "km", "SettleDist": "km", "PopDensity": "persons/km²", "Elevation": "m",
         "Slope": "°", "CBDDist": "km", "InfraProx": "km"}


def get_real_df():
    """Load ONLY the real dataset. Never falls back to placeholder data."""
    if not os.path.exists(CSV_PATH):
        return None, f"File not found: {CSV_PATH}"
    d = pd.read_csv(CSV_PATH)
    need = ["lat", "lon"] + FEATURES + [TARGET]
    miss = [c for c in need if c not in d.columns]
    if miss:
        return None, f"The dataset is missing columns: {', '.join(miss)}"
    d = d.dropna(subset=need)
    if len(d) < 200 or d[TARGET].nunique() < 2 or int(d[TARGET].sum()) < 20:
        return None, "The dataset is too small, or has too few expansion cells, to train on."
    return d, None


@st.cache_resource
def train(df):
    X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df[TARGET], test_size=0.2,
                                              stratify=df[TARGET], random_state=42)
    sc = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(X_tr), y_tr)
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42).fit(X_tr, y_tr)
    return sc, lr, rf, X_te, y_te


@st.cache_resource(ttl=3600)  # refreshed hourly so the 'latest' imagery window stays current
def get_ee(project=None):
    """Initialise Google Earth Engine from Streamlit secrets, a typed project id, or a local `earthengine authenticate`."""
    try:
        sa = dict(st.secrets["gee_service_account"]) if "gee_service_account" in st.secrets else None
        project = project or (st.secrets["gee_project"] if "gee_project" in st.secrets else None) or os.environ.get("GEE_PROJECT")
        ee = ft.init_ee(sa, project)
        ee.Number(1).getInfo()  # fails fast if the account/project is not allowed to use Earth Engine
    except Exception as e:
        return None, {}, f"{type(e).__name__}: {e}"
    try:
        layers = ft.ee_tile_urls(ee)
    except Exception:
        layers = {}
    return ee, layers, None


def cached_extract(lat, lon, use_ee):
    """Remember results for this click in the session (also when some lookups failed; use Retry to redo)."""
    key = f"ext_{lat}_{lon}"
    if key not in st.session_state:
        st.session_state[key] = ft.extract_all(lat, lon, get_ee(st.session_state.get("gee_project"))[0] if use_ee else None, _built_year())
    return st.session_state[key]


@st.cache_data(show_spinner=False, ttl=300)
def cached_name(lat, lon):
    return ft.reverse_geocode(lat, lon)


STEP = {"RoadDist": 0.1, "SettleDist": 0.1, "PopDensity": 100.0, "Elevation": 10.0,
        "Slope": 1.0, "CBDDist": 0.1, "InfraProx": 0.1}


def predict(name, X):
    X = pd.DataFrame(X, columns=FEATURES)
    return lr.predict_proba(sc.transform(X))[:, 1] if name == "Logistic Regression" else rf.predict_proba(X)[:, 1]


def zone(p):
    return ("Critical high-risk" if p > 0.85 else "High" if p > 0.70 else "Moderate buffer" if p >= 0.30
            else "Low" if p >= 0.10 else "Very low / protected")


def risk_color(p):
    safe, mid, risk = (63, 117, 102), (201, 138, 58), (161, 59, 42)
    lo, hi, t = (safe, mid, p / 0.5) if p < 0.5 else (mid, risk, (p - 0.5) / 0.5)
    return "rgb(%d,%d,%d)" % tuple(round(x + (y - x) * t) for x, y in zip(lo, hi))


def prob_card(label, p, delta):
    c = risk_color(p)
    return (f'<div style="border:1px solid #8884;border-radius:8px;padding:12px 14px;margin-bottom:10px">'
            f'<div style="font-size:12px;opacity:.7">{label}</div>'
            f'<div style="font-size:36px;font-weight:700;color:{c};line-height:1.1">{p:.2%}</div>'
            f'<div style="height:8px;border-radius:4px;background:#8883;margin:8px 0">'
            f'<div style="height:100%;width:{p * 100:.1f}%;background:{c};border-radius:4px"></div></div>'
            f'<div style="font-size:12px;opacity:.7">{delta:+.2f} pts vs starting values</div></div>')


FMT = {"RoadDist": "%.2f", "SettleDist": "%.2f", "PopDensity": "%.0f", "Elevation": "%.0f",
       "Slope": "%.1f", "CBDDist": "%.2f", "InfraProx": "%.2f"}

def location_map():
    """Map with a 'use my location' button. Stores the chosen point in st.session_state.sel. Returns informal nodes."""
    if streamlit_geolocation is not None:
        g1, g2 = st.columns([1, 4])
        with g1:
            loc = streamlit_geolocation()
        with g2:
            st.caption("Press the target button, then allow location access in your browser to use your current position. "
                       "On a PC the position comes from Wi-Fi/IP and can be off by a kilometre or more.")
        if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
            pos = (round(float(loc["latitude"]), 5), round(float(loc["longitude"]), 5))
            if pos != st.session_state.get("gps"):  # react only to a NEW reading
                st.session_state.gps, st.session_state.gps_acc, st.session_state.sel = pos, loc.get("accuracy"), pos
                st.rerun()
    center = st.session_state.get("sel", (-22.56, 17.08))
    m = folium.Map(location=list(center), zoom_start=14 if "sel" in st.session_state else 11, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Imagery", name="Esri satellite (high detail, not live)").add_to(m)
    day = (date.today() - timedelta(days=1)).isoformat()  # GIBS publishes yesterday's daily mosaic
    folium.TileLayer(
        tiles=("https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/"
               f"{day}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg"),
        attr="NASA GIBS / MODIS", name=f"NASA daily satellite ({day}, 250 m, near real-time)",
        max_native_zoom=9, max_zoom=18).add_to(m)
    for name, url in ee_layers.items():
        folium.TileLayer(tiles=url, attr="Google Earth Engine", name=name, overlay=True, show=False).add_to(m)
    nodes, _ = ft.load_nodes()
    for n, (la, lo) in nodes.items():
        folium.CircleMarker((la, lo), radius=5, color="purple", fill=True, tooltip=f"Informal settlement: {n}").add_to(m)
    folium.Marker(ft.CBD, tooltip="CBD reference", icon=folium.Icon(color="black", icon="star")).add_to(m)
    if "sel" in st.session_state:
        folium.Marker(st.session_state.sel, tooltip="Selected", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl().add_to(m)
    out = st_folium(m, height=480, use_container_width=True, returned_objects=["last_clicked"])
    if out and out.get("last_clicked"):
        new = (round(out["last_clicked"]["lat"], 5), round(out["last_clicked"]["lng"], 5))
        if new != st.session_state.get("sel"):
            st.session_state.sel = new
            st.rerun()
    sel = st.session_state.get("sel")
    if sel and st.session_state.get("gps") == sel and st.session_state.get("gps_acc"):
        st.caption(f"Using your device location (accuracy about +/-{float(st.session_state.gps_acc):.0f} m)")
    if sel and not (LAT0 <= sel[0] <= LAT1 and LON0 <= sel[1] <= LON1):
        st.warning("This point is outside the study grid (lat -22.65 to -22.45, lon 17.00 to 17.18). "
                   "The models were not trained here, so any probability would be an extrapolation.")
    return nodes


def no_data_screen(problem):
    #st.error("No real dataset has been loaded yet, so the models are NOT trained and no probabilities are shown. "
     #        "This app never trains on placeholder data. Build the real dataset from Google Earth Engine below.")
    st.caption(problem)
    st.subheader("Build the real dataset from Google Earth Engine")
    proj = st.text_input("Google Cloud project ID (the one with Earth Engine enabled)",
                         value=st.session_state.get("gee_project", ""))
    k1, k2, k3 = st.columns(3)
    t0 = k1.number_input("Start year (predictors)", 2016, 2023, 2018)
    t1 = k2.number_input("End year (label)", 2019, 2026, 2025)
    mn = k3.number_input("Min new built-up share of a cell for Y = 1", 0.01, 0.50, 0.05, 0.01)
    if st.button("Connect to Earth Engine"):
        st.session_state.gee_project = proj.strip()
        get_ee.clear()
        st.rerun()
    if ee is None:
        st.warning("Not connected to Earth Engine yet." + (f" ({ee_err[:200]})" if ee_err else ""))
        st.markdown("**One-time setup:** 1) register at earthengine.google.com (free for research), "
                    "2) create a Google Cloud project with the Earth Engine API enabled, "
                    "3) stop this app, run `earthengine authenticate` in the terminal and sign in, "
                    "4) restart the app, type the project ID above and press Connect.")
    else:
        st.success("Connected to Google Earth Engine.")
        if st.button("Build dataset and train", type="primary"):
            with st.status("Building the dataset (about 5-20 minutes). Keep this tab open.", expanded=True) as status:
                try:
                    polys = bd.load_informal(log=st.write)
                    st.write("Label definition: " + ("new built-up inside the informal-settlement footprint" if polys
                             else "ALL new built-up growth (no informal-settlement layer found)"))
                    df_new, meta = bd.build(ee, int(t0), int(t1), float(mn), polys, log=st.write)
                    bd.save(df_new, meta, log=st.write)
                    status.update(label="Dataset built - training the models next", state="complete")
                except Exception as e:
                    status.update(label="Build failed", state="error")
                    st.exception(e)
                    return
            st.rerun()
    with st.expander("Optional: add your own informal-settlement polygons for a more specific label"):
        st.markdown("Save polygons as `data/informal_settlements.geojson` before pressing Build. "
                    "If you already have your own CSV, save it as `data/windhoek_grid.csv` with columns "
                    "`lat, lon, RoadDist, SettleDist, PopDensity, Elevation, Slope, CBDDist, InfraProx, Expansion`.")
    st.subheader("Location finder (works without the models)")
    location_map()
    sel = st.session_state.get("sel")
    if sel:
        lat, lon = sel
        st.markdown(f"**{cached_name(lat, lon) or 'Selected location'}**  ({lat:.5f}, {lon:.5f})")
        with st.spinner("Extracting predictors..."):
            vals, notes = cached_extract(lat, lon, ee is not None)
        st.dataframe(pd.DataFrame({"Predictor": FEATURES,
                                   "Value": [None if vals.get(f) is None else round(float(vals[f]), 3) for f in FEATURES],
                                   "Unit": [UNITS[f] for f in FEATURES], "Source": [notes.get(f, "") for f in FEATURES]}),
                     hide_index=True)
    else:
        st.info("Press the location button or click the map to see all 7 predictors for a place.")


df, data_problem = get_real_df()
ee, ee_layers, ee_err = get_ee(st.session_state.get("gee_project"))

st.title("A Data-Driven Framework for Predicting Informal Settlement Expansion in Windhoek "
         "Using Spatial Analysis and Machine Learning Techniques")
st.markdown("**Witbooi Hamunyela**  |  B.Sc Data Science (Honours), University of Namibia")
st.caption("Department of Computing, Mathematical and Statistical Sciences, Faculty of Agriculture, "
           "Engineering & Natural Science  |  Research prototype")
st.divider()
st.sidebar.caption("Google Earth Engine: " + ("connected" if ee else "not connected (OpenTopoData fallback; "
                   "PopDensity must be entered manually)"))
if ee_err:
    st.sidebar.caption(f"GEE error: {ee_err[:160]}")
if df is None:
    no_data_screen(data_problem)
    st.stop()
formal_geoms = ft.load_formal()
df_all = df
keep = pd.Series(True, index=df.index)
if "built_t0_frac" in df.columns:
    keep &= df["built_t0_frac"] < MAX_BUILT
if formal_geoms:
    keep &= ~pd.Series([ft.point_in_geoms(la, lo, formal_geoms) for la, lo in zip(df.lat, df.lon)], index=df.index)
if keep.sum() >= 200 and int(df.loc[keep, TARGET].sum()) >= 20:
    df = df[keep].reset_index(drop=True)
else:
    keep = pd.Series(True, index=df_all.index)
sc, lr, rf, X_te, y_te = train(df)
st.sidebar.caption(f"{int((~keep).sum())} cells excluded from training and prediction: already developed "
                   f"(>= {MAX_BUILT:.0%} built-up)" + (" or inside the formal areas you marked." if formal_geoms else "."))
model_name = st.sidebar.radio("Model", ["Random Forest", "Logistic Regression"])
st.sidebar.success(f"Trained on real data: {len(df)} cells, {int(df[TARGET].sum())} expansion ({df[TARGET].mean():.1%})")
_meta = os.path.join(os.path.dirname(CSV_PATH), "dataset_metadata.json")
if os.path.exists(_meta):
    import json as _json
    _m = _json.load(open(_meta))
    st.sidebar.caption(f"Label: {_m.get('label_definition', '?')} ({_m.get('t0')} to {_m.get('t1')}). "
                       "Source: Google Earth Engine + OpenStreetMap.")

tab_map, tab_grid, tab_perf, tab_eda = st.tabs(["Pick a location", "Grid risk map", "Model performance", "Explore data"])

with tab_map:
    summary = st.container()  # filled in once the values are known, so the probability is always at the top
    st.write("Click anywhere on the map. All 7 predictors are extracted for that point, "
             "and you can edit any value before the models score it.")
    nodes = location_map()

    sel = st.session_state.get("sel")
    if sel:
        lat, lon = sel
        place = cached_name(lat, lon)
        near, near_d = min(((n, ft.haversine_km(lat, lon, *c)) for n, c in nodes.items()), key=lambda x: x[1])
        heading = place or "Place name not found (OpenStreetMap has no name here)"
        st.subheader(heading)
        st.caption(f"Coordinates {lat:.5f}, {lon:.5f}  |  nearest known informal settlement: {near} ({near_d:.1f} km)")
        with st.spinner("Extracting predictors from Earth Engine and OpenStreetMap..."):
            vals, notes = cached_extract(lat, lon, ee is not None)
    else:
        lat = lon = "example"
        heading = "Example parcel (dataset typical values)"
        vals, notes = {}, {}
        st.subheader(heading)
        st.info("No location clicked yet. This example parcel shows how the probability works. "
                "Click the map to score a real location.")
    base = {f: float(df[f].median()) if vals.get(f) is None else float(vals[f]) for f in FEATURES}
    reason = None
    if sel:
        built = vals.get("BuiltFrac")
        if ft.point_in_geoms(lat, lon, formal_geoms):
            reason = "this point is inside a formal (planned) area"
        elif built is not None and built >= MAX_BUILT:
            reason = f"{built:.0%} of the land within 200 m is already built-up"

    ck = f"cur_{lat}_{lon}"  # the current values for this location, kept apart from widget state
    st.session_state.setdefault(ck, dict(base))

    def set_cur(row):
        st.session_state[ck] = {f: float(row[f]) for f in FEATURES}
        for f in FEATURES:
            for kind in ("n", "s"):
                st.session_state.pop(f"{kind}_{f}_{lat}_{lon}", None)

    def retry():
        st.session_state.pop(f"ext_{lat}_{lon}", None)
        st.session_state.pop(ck, None)

    def lim(f):
        return min(float(df[f].min()), base[f]), max(float(df[f].max()), base[f])

    presets = {
        "Dataset typical": df[FEATURES].median(),
        "Valley floor": df[df[TARGET] == 1][FEATURES].median(),
        "Hillside": df[df.Slope >= df.Slope.quantile(0.75)][FEATURES].median(),
        "Near CBD": df[df.CBDDist <= df.CBDDist.quantile(0.10)][FEATURES].median(),
    }
    mode = st.radio("Edit values with", ["Sliders", "Number boxes"], horizontal=True, key="edit_mode")
    st.caption("Presets use typical values from the dataset:")
    for col, (name, row) in zip(st.columns(4), presets.items()):
        col.button(name, on_click=set_cur, args=(row,), key=f"preset_{name}", width="stretch")
    b1, b2 = st.columns(2)
    b1.button("Reset to extracted values", on_click=set_cur, args=(base,), width="stretch")
    b2.button("Retry lookups", on_click=retry, width="stretch")
    if sel:
        got = sum(vals.get(f) is not None for f in FEATURES)
        st.caption(f"{got} of 7 predictors were extracted automatically; the rest start at the dataset median.")

    cur, used, grid_cols = st.session_state[ck], {}, st.columns(2)
    for i, f in enumerate(FEATURES):
        lo, hi = lim(f)
        label = f"{f} ({UNITS[f]})"
        with grid_cols[i % 2]:
            if mode == "Sliders":
                k = f"s_{f}_{lat}_{lon}"
                if k not in st.session_state:
                    st.session_state[k] = float(min(max(cur[f], lo), hi))
                used[f] = st.slider(label, lo, hi, step=STEP[f], format=FMT[f], key=k)
            else:
                k = f"n_{f}_{lat}_{lon}"
                if k not in st.session_state:
                    st.session_state[k] = float(cur[f])
                used[f] = st.number_input(label, step=STEP[f], format="%.3f", key=k)
            if sel:
                st.caption(notes.get(f, "") + ("" if vals.get(f) is not None else "  [using dataset median]"))
    st.session_state[ck] = dict(used)

    p_lr, p_rf = (float(predict(n, [used])[0]) for n in ("Logistic Regression", "Random Forest"))
    b_lr, b_rf = (float(predict(n, [base])[0]) for n in ("Logistic Regression", "Random Forest"))
    sel_p = p_rf if model_name == "Random Forest" else p_lr

    def show_cards():
        c1, c2, c3 = st.columns(3)
        c1.markdown(prob_card("Probability of expansion (Logistic Regression)", p_lr, (p_lr - b_lr) * 100),
                    unsafe_allow_html=True)
        c2.markdown(prob_card("Probability of expansion (Random Forest)", p_rf, (p_rf - b_rf) * 100),
                    unsafe_allow_html=True)
        c3.metric(f"Risk zone ({model_name})", zone(sel_p))

    with summary:  # appears at the very top of the tab
        st.markdown("### Probability of expansion")
        st.caption(heading)
        if reason:
            st.warning(f"Informal expansion is not possible here: {reason}. The models are not applied to developed or formal land.")
            with st.expander("Show the raw model output anyway (not meaningful for this location)"):
                show_cards()
        else:
            show_cards()

    if max(p_lr, p_rf) < 0.01 or min(p_lr, p_rf) > 0.99:
        st.info("The models are saturated here: slope (and elevation) dominate, so small changes to the other "
                "variables barely move the probability. Try changing Slope, or see the curves below.")

    z = sc.transform(pd.DataFrame([used], columns=FEATURES))[0]
    contrib = pd.DataFrame({"feature": FEATURES, "logit contribution": lr.coef_[0] * z}).sort_values("logit contribution")
    st.plotly_chart(px.bar(contrib, x="logit contribution", y="feature", orientation="h",
                           title="What pushes this location up or down (logistic regression)"), width="stretch")

    st.markdown("**What-if curve:** how the probability changes as one variable moves, others held at the values above")
    sf = st.selectbox("Variable to vary", FEATURES, index=FEATURES.index("Slope"))
    grid = np.linspace(df[sf].quantile(0.01), df[sf].quantile(0.99), 60)
    sweep = pd.DataFrame([{**used, sf: g} for g in grid], columns=FEATURES)
    curve = pd.DataFrame({sf: grid, "Logistic Regression": predict("Logistic Regression", sweep),
                          "Random Forest": predict("Random Forest", sweep)}).melt(sf, var_name="Model", value_name="P(expansion)")
    fig = px.line(curve, x=sf, y="P(expansion)", color="Model", range_y=(0, 1))
    fig.add_vline(x=used[sf], line_dash="dash")
    st.plotly_chart(fig, width="stretch")

with tab_grid:
    d = df.copy()
    d["risk"] = predict(model_name, d[FEATURES])
    fig = px.scatter(d, x="lon", y="lat", color="risk", range_color=(0, 1), color_continuous_scale="YlOrRd", height=600)
    out_cells = df_all[~keep]
    if len(out_cells):
        fig.add_scatter(x=out_cells.lon, y=out_cells.lat, mode="markers", marker=dict(color="#cfcfcf", size=7),
                        name="Already developed / formal (not modelled)")
    fig.update_yaxes(scaleanchor="x")
    st.plotly_chart(fig, width="stretch")

with tab_perf:
    rows = []
    for n in ["Logistic Regression", "Random Forest"]:
        pr = predict(n, X_te)
        yh = (pr >= 0.5).astype(int)
        rows.append({"Model": n, "Accuracy": accuracy_score(y_te, yh), "Precision": precision_score(y_te, yh),
                     "Recall": recall_score(y_te, yh), "F1": f1_score(y_te, yh), "ROC-AUC": roc_auc_score(y_te, pr)})
    st.dataframe(pd.DataFrame(rows).set_index("Model").round(3))
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    st.plotly_chart(px.bar(imp, orientation="h", labels={"value": "Gini importance", "index": ""}), width="stretch")

with tab_eda:
    st.dataframe(df.describe().T.round(2))
    f = st.selectbox("Predictor", FEATURES)
    st.plotly_chart(px.box(df, x=TARGET, y=f, color=TARGET), width="stretch")
