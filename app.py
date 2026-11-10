"""Windhoek Informal Settlement Expansion Risk - Streamlit app with an interactive map."""
import os

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

import features as ft
from data_prep import CSV_PATH, FEATURES, TARGET, load_data

st.set_page_config(page_title="Informal Settlement Expansion Prediction - Windhoek", page_icon="🏘️", layout="wide")
UNITS = {"RoadDist": "km", "SettleDist": "km", "PopDensity": "persons/km²", "Elevation": "m",
         "Slope": "°", "CBDDist": "km", "InfraProx": "km"}


@st.cache_data
def get_df():
    return load_data()


@st.cache_resource
def train(df):
    X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df[TARGET], test_size=0.2,
                                              stratify=df[TARGET], random_state=42)
    sc = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(X_tr), y_tr)
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42).fit(X_tr, y_tr)
    return sc, lr, rf, X_te, y_te


@st.cache_resource
def get_ee():
    """Initialise Google Earth Engine from Streamlit secrets (or a local `earthengine authenticate`)."""
    try:
        sa = dict(st.secrets["gee_service_account"]) if "gee_service_account" in st.secrets else None
        project = st.secrets["gee_project"] if "gee_project" in st.secrets else None
        ee = ft.init_ee(sa, project)
        return ee, ft.ee_tile_urls(ee), None
    except Exception as e:
        return None, {}, f"{type(e).__name__}: {e}"


def cached_extract(lat, lon, use_ee):
    """Remember results for this click in the session (also when some lookups failed; use Retry to redo)."""
    key = f"ext_{lat}_{lon}"
    if key not in st.session_state:
        st.session_state[key] = ft.extract_all(lat, lon, get_ee()[0] if use_ee else None)
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

df = get_df()
sc, lr, rf, X_te, y_te = train(df)
ee, ee_layers, ee_err = get_ee()

st.title("A Data-Driven Framework for Predicting Informal Settlement Expansion in Windhoek "
         "Using Spatial Analysis and Machine Learning Techniques")
st.markdown("**Witbooi Hamunyela**  |  B.Sc Data Science (Honours), University of Namibia")
st.caption("Department of Computing, Mathematical and Statistical Sciences, Faculty of Agriculture, "
           "Engineering & Natural Science  |  Research prototype")
st.divider()
if not os.path.exists(CSV_PATH):
    st.warning("Models are trained on a PLACEHOLDER dataset. Add data/windhoek_grid.csv for real results.")
model_name = st.sidebar.radio("Model", ["Random Forest", "Logistic Regression"])
st.sidebar.caption("Google Earth Engine: " + ("connected" if ee else "not connected (OpenTopoData fallback; "
                   "PopDensity must be entered manually)"))
if ee_err:
    st.sidebar.caption(f"GEE error: {ee_err[:160]}")

tab_map, tab_grid, tab_perf, tab_eda = st.tabs(["Pick a location", "Grid risk map", "Model performance", "Explore data"])

with tab_map:
    st.write("Click anywhere on the map. All 7 predictors are extracted for that point, "
             "and you can edit any value before the models score it.")
    m = folium.Map(location=[-22.56, 17.08], zoom_start=11, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Imagery", name="Esri satellite").add_to(m)
    for name, url in ee_layers.items():
        folium.TileLayer(tiles=url, attr="Google Earth Engine", name=name, overlay=True, show=False).add_to(m)
    nodes, _ = ft.load_nodes()
    for n, (la, lo) in nodes.items():
        folium.CircleMarker((la, lo), radius=5, color="purple", fill=True, tooltip=f"Informal settlement: {n}").add_to(m)
    folium.Marker(ft.CBD, tooltip="CBD reference", icon=folium.Icon(color="black", icon="star")).add_to(m)
    if "sel" in st.session_state:
        folium.Marker(st.session_state.sel, tooltip="Selected", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl().add_to(m)
    out = st_folium(m, height=520, use_container_width=True, returned_objects=["last_clicked"])
    if out and out.get("last_clicked"):
        new = (round(out["last_clicked"]["lat"], 5), round(out["last_clicked"]["lng"], 5))
        if new != st.session_state.get("sel"):
            st.session_state.sel = new
            st.rerun()

    if "sel" in st.session_state:
        lat, lon = st.session_state.sel
        place = cached_name(lat, lon)
        near, near_d = min(((n, ft.haversine_km(lat, lon, *c)) for n, c in nodes.items()), key=lambda x: x[1])
        st.subheader(place or "Place name not found (OpenStreetMap has no name here)")
        st.caption(f"Coordinates {lat:.5f}, {lon:.5f}  |  nearest known informal settlement: {near} ({near_d:.1f} km)")
        with st.spinner("Extracting predictors from Earth Engine and OpenStreetMap..."):
            vals, notes = cached_extract(lat, lon, ee is not None)
        base = {f: float(df[f].median()) if vals.get(f) is None else float(vals[f]) for f in FEATURES}

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
        left, right = st.columns([3, 2])
        used = {}
        with left:
            mode = st.radio("Edit values with", ["Sliders", "Number boxes"], horizontal=True, key="edit_mode")
            st.caption("Presets use typical values from the dataset:")
            for col, (name, row) in zip(st.columns(4), presets.items()):
                col.button(name, on_click=set_cur, args=(row,), key=f"preset_{name}", width="stretch")
            b1, b2 = st.columns(2)
            b1.button("Reset to extracted values", on_click=set_cur, args=(base,), width="stretch")
            b2.button("Retry lookups", on_click=retry, width="stretch")
            got = sum(vals.get(f) is not None for f in FEATURES)
            st.caption(f"{got} of 7 predictors were extracted automatically; the rest start at the dataset median.")
            cur = st.session_state[ck]
            for f in FEATURES:
                lo, hi = lim(f)
                label = f"{f} ({UNITS[f]})"
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
                st.caption(notes.get(f, "") + ("" if vals.get(f) is not None else "  [using dataset median]"))
            st.session_state[ck] = dict(used)

        p_lr, p_rf = (float(predict(n, [used])[0]) for n in ("Logistic Regression", "Random Forest"))
        b_lr, b_rf = (float(predict(n, [base])[0]) for n in ("Logistic Regression", "Random Forest"))
        sel_p = p_rf if model_name == "Random Forest" else p_lr
        with right:
            st.markdown("### Probability of expansion at this location")
            st.markdown(prob_card("Probability of expansion (Logistic Regression)", p_lr, (p_lr - b_lr) * 100),
                        unsafe_allow_html=True)
            st.markdown(prob_card("Probability of expansion (Random Forest)", p_rf, (p_rf - b_rf) * 100),
                        unsafe_allow_html=True)
            st.markdown(f"**Risk zone ({model_name}):** {zone(sel_p)}")
        if not os.path.exists(CSV_PATH):
            st.caption("Prototype: models are trained on placeholder data, so treat these probabilities as illustrative.")
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
    else:
        st.info("No location selected yet.")

with tab_grid:
    d = df.copy()
    d["risk"] = predict(model_name, d[FEATURES])
    fig = px.scatter(d, x="lon", y="lat", color="risk", range_color=(0, 1), color_continuous_scale="YlOrRd", height=600)
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
