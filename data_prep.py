"""Step 1 - Build the Windhoek grid dataset (2,500 cells, 7 predictors, binary label).

If you have real data, save it as data/windhoek_grid.csv with columns:
lat, lon, RoadDist, SettleDist, PopDensity, Elevation, Slope, CBDDist, InfraProx, Expansion
Otherwise this script builds a PLACEHOLDER grid so Steps 2-6 can be developed now
and re-run on real data later without changing any code.
"""
import os
import numpy as np
import pandas as pd

FEATURES = ["RoadDist", "SettleDist", "PopDensity", "Elevation", "Slope", "CBDDist", "InfraProx"]
TARGET = "Expansion"
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "windhoek_grid.csv")

# Table 4.1 of the thesis, used to sanity-check the placeholder
TABLE_4_1 = {
    "RoadDist": (2.839, 2.366), "SettleDist": (8.086, 4.455), "PopDensity": (1317.0, 3331.4),
    "Elevation": (1769.6, 130.4), "Slope": (25.70, 9.39), "CBDDist": (8.074, 3.164),
    "InfraProx": (5.999, 3.239),
}


def make_placeholder(seed=42, n_side=50, prevalence=0.166):
    rng = np.random.default_rng(seed)
    lat = np.linspace(-22.65, -22.45, n_side)
    lon = np.linspace(17.00, 17.18, n_side)
    LON, LAT = np.meshgrid(lon, lat)
    df = pd.DataFrame({"lat": LAT.ravel(), "lon": LON.ravel()})
    n = len(df)

    # km distances (1 deg lat ~ 111 km; 1 deg lon ~ 102.5 km at 22.6 S)
    def km(lat0, lon0):
        return np.hypot((df.lat - lat0) * 111.0, (df.lon - lon0) * 102.5)

    df["CBDDist"] = km(-22.565, 17.083)
    nodes = [(-22.50, 17.03), (-22.54, 17.03), (-22.47, 17.05), (-22.58, 17.03)]
    df["SettleDist"] = np.min([km(a, b) for a, b in nodes], axis=0) + rng.gamma(1.5, 1.0, n)
    df["RoadDist"] = np.clip(rng.gamma(1.8, 1.6, n), 0.02, 10.3)
    ridge = (df.lon - 17.0) / 0.18  # terrain rises eastward (Auas / Eros)
    df["Elevation"] = np.clip(1600 + 330 * ridge + rng.normal(0, 40, n), 1579, 2147)
    df["Slope"] = np.clip(5 + 0.12 * (df.Elevation - 1600) + rng.normal(0, 6, n), 0.78, 35)
    df["PopDensity"] = np.clip(np.exp(rng.normal(8.0 - 0.5 * df.SettleDist, 1.5)), 0.1, 19540)
    df["InfraProx"] = np.clip(0.5 * df.CBDDist + rng.gamma(2, 1.0, n), 0.09, 15.5)

    # Placeholder label: logistic on the thesis betas (z-scored), intercept tuned to ~16.6%
    betas = dict(RoadDist=-1.17, SettleDist=-0.96, PopDensity=0.40, Elevation=-1.99,
                 Slope=-3.92, CBDDist=0.32, InfraProx=1.09)
    z = sum(b * (df[f] - df[f].mean()) / df[f].std() for f, b in betas.items())
    lo, hi = -20.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (1 / (1 + np.exp(-(z + mid)))).mean() < prevalence else (lo, mid)
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-(z + mid)))).astype(int)
    return df


def load_data():
    if os.path.exists(CSV_PATH):
        print(f"Loading real data from {CSV_PATH}")
        return pd.read_csv(CSV_PATH)
    print("No real data found -> building PLACEHOLDER grid (not for reporting results)")
    return make_placeholder()


def check(df):
    print(f"\nRows: {len(df)} | Expansion rate: {df[TARGET].mean():.1%} (thesis: 16.6%)")
    print(f"Missing values: {int(df[FEATURES].isna().sum().sum())}")
    print(f"\n{'Feature':<11}{'mean':>9}{'(thesis)':>10}{'sd':>9}{'(thesis)':>10}")
    for f in FEATURES:
        m, s = TABLE_4_1[f]
        print(f"{f:<11}{df[f].mean():>9.2f}{m:>10.2f}{df[f].std():>9.2f}{s:>10.2f}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = load_data()
    check(df)
    df.to_csv("data/grid_clean.csv", index=False)
    print("\nSaved data/grid_clean.csv")
