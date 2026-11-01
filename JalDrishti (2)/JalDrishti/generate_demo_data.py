"""Generate a SYNTHETIC demo dataset so the pipeline can run end to end.

!!! This data is artificial. Models trained on it say NOTHING about real groundwater.
Replace it with real, documented well observations (data/wells.csv).

Study area: central Malwa (roughly Ratlam - Indore - Dewas - Ujjain).
The synthetic "true" water potential has a smooth regional pattern (wet and dry zones),
so results differ from place to place. The nearby_well_* features are computed from the
OTHER wells around each well (never the well itself), exactly as the app does at prediction time.
"""
import numpy as np
import pandas as pd
from config import DATA_DIR, DEMO_DATA_PATH, RANDOM_STATE
import geo_features as G

LAT_RANGE = (22.6, 23.8)
LON_RANGE = (74.9, 76.3)


def make_demo(n: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    lat = rng.uniform(*LAT_RANGE, n)
    lon = rng.uniform(*LON_RANGE, n)
    rainfall = rng.normal(900, 70, n) + (lon - 74.9) * 70 + (lat - 22.6) * 40
    elevation = rng.normal(500, 25, n) + (lat - 22.6) * 60
    slope = np.clip(rng.gamma(2.0, 1.8, n), 0, 25)
    dist_stream = np.clip(rng.exponential(700, n), 10, 4000)
    drainage_density = np.clip(rng.normal(1.6, 0.5, n), 0.2, 4)
    ndvi = np.clip(rng.normal(0.45, 0.15, n), 0, 0.9)
    soil = rng.choice(["black_cotton", "red_loam", "alluvial", "sandy"], n, p=[.45, .25, .2, .1])
    geology = rng.choice(["deccan_basalt", "alluvium", "granite", "sandstone"], n, p=[.55, .15, .15, .15])
    landcover = rng.choice(["agriculture", "built_up", "forest", "scrub"], n, p=[.65, .1, .1, .15])

    geo_bonus = pd.Series(geology).map(
        {"deccan_basalt": 0.0, "alluvium": 0.9, "granite": -0.3, "sandstone": 0.4}).values
    soil_bonus = pd.Series(soil).map(
        {"black_cotton": -0.1, "red_loam": 0.1, "alluvial": 0.5, "sandy": 0.3}).values
    # smooth regional pattern: wet and dry zones across the map
    region = 2.4 * np.sin(3.6 * (lat - 22.6)) * np.cos(3.0 * (lon - 74.9))
    z = (-0.2 + 0.004 * (rainfall - 900) - 0.12 * slope - 0.0006 * dist_stream
         - 0.35 * drainage_density + region + geo_bonus + soil_bonus + rng.normal(0, 0.6, n))
    label = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int)

    well_depth = np.where(label == 1, rng.normal(110, 30, n), rng.normal(160, 40, n)).round(1)
    water_level = np.where(label == 1, rng.normal(25, 8, n), np.nan).round(1)

    df = pd.DataFrame(dict(
        lat=lat.round(5), lon=lon.round(5), rainfall=rainfall.round(0),
        elevation=elevation.round(0), slope=slope.round(2), dist_stream=dist_stream.round(0),
        drainage_density=drainage_density.round(2), ndvi=ndvi.round(3),
        soil=soil, geology=geology, landcover=landcover,
        well_depth=well_depth, water_level=water_level, label=label))

    # Neighbour features from the OTHER wells only (no self-leakage)
    succ, depth = [], []
    for i in range(n):
        idx, _ = G.nearest_wells(df, df.lat[i], df.lon[i], exclude_index=i)
        if len(idx) == 0:
            succ.append(np.nan); depth.append(np.nan)
        else:
            succ.append(df.label.iloc[idx].mean()); depth.append(df.well_depth.iloc[idx].mean())
    df["nearby_well_success_rate"] = np.round(succ, 2)
    df["nearby_well_avg_depth"] = np.round(depth, 0)
    return df


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    df = make_demo()
    df.to_csv(DEMO_DATA_PATH, index=False)
    print(f"Wrote {len(df)} SYNTHETIC rows to {DEMO_DATA_PATH}")
    print(df.label.value_counts(normalize=True).round(2).to_dict())
