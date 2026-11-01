"""Address -> coordinates, and coordinates -> model features.

geocode(): OpenStreetMap Nominatim via geopy (free; max 1 request/second).

get_features(): builds the model inputs for a point from the WELL RECORDS around it
(data/wells.csv if present, else the synthetic demo file):
  * nearby_well_success_rate = share of the nearest wells that found water
  * nearby_well_avg_depth    = average depth of the nearest wells
  * every other feature      = distance-weighted average (or weighted vote for categories)
                               of the nearest wells' values
If there are no wells within MAX_KM, it falls back to study-area medians and says so.

The best upgrade is to replace the "other features" part with real raster / shapefile
lookups (DEM, rainfall, geology, soil) at the exact point; see the TODO below.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib

import config as C

K_NEIGHBOURS = 8      # how many nearby wells to use
MAX_KM = 25.0         # ignore wells farther than this
MIN_NEIGHBOURS = 3    # need at least this many wells nearby


@lru_cache(maxsize=256)
def geocode(address: str):
    from geopy.geocoders import Nominatim
    geo = Nominatim(user_agent="jaldrishti-student-project")
    loc = geo.geocode(address, timeout=10)
    if loc is None:
        return None
    return float(loc.latitude), float(loc.longitude), loc.address


@lru_cache(maxsize=1)
def load_wells():
    path = C.REAL_DATA_PATH if C.REAL_DATA_PATH.exists() else C.DEMO_DATA_PATH
    if not path.exists():
        return None
    df = pd.read_csv(path).dropna(subset=["lat", "lon", C.TARGET]).reset_index(drop=True)
    return df


def _haversine_km(lat, lon, lats, lons):
    p = np.pi / 180.0
    dlat, dlon = (lats - lat) * p, (lons - lon) * p
    a = np.sin(dlat / 2) ** 2 + np.cos(lat * p) * np.cos(lats * p) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def nearest_wells(wells, lat, lon, exclude_index=None):
    """Indices and distances (km) of the nearest wells within MAX_KM."""
    d = _haversine_km(lat, lon, wells.lat.values, wells.lon.values)
    if exclude_index is not None:          # never let a well "see" itself
        d[exclude_index] = np.inf
    idx = np.argsort(d)[:K_NEIGHBOURS]
    keep = d[idx] <= MAX_KM
    return idx[keep], d[idx][keep]


def features_from_wells(wells, lat, lon, exclude_index=None):
    """Return (feature_dict, info) where info says how the features were obtained."""
    idx, dist = nearest_wells(wells, lat, lon, exclude_index)
    if len(idx) < MIN_NEIGHBOURS:
        return None, {"source": "placeholder", "n_neighbors": int(len(idx)), "nearest_km": None}
    near = wells.iloc[idx]
    w = 1.0 / (dist + 0.5) ** 2
    w = w / w.sum()

    feats = {}
    for f in C.NUMERIC_FEATURES:
        if f == "nearby_well_success_rate":
            feats[f] = float(near[C.TARGET].mean())
        elif f == "nearby_well_avg_depth" and "well_depth" in near and near["well_depth"].notna().any():
            feats[f] = float(near["well_depth"].mean())
        elif f in near and near[f].notna().any():
            ok = near[f].notna().values
            feats[f] = float(np.average(near[f].values[ok], weights=w[ok]))
    for f in C.CATEGORICAL_FEATURES:
        if f in near and near[f].notna().any():
            votes = pd.Series(w, index=near[f].values)[near[f].notna().values]
            feats[f] = votes.groupby(level=0).sum().idxmax()
    # TODO (best upgrade): overwrite elevation, slope, rainfall, geology, soil, landcover,
    # dist_stream, drainage_density, ndvi with exact values sampled from rasters/shapefiles
    # at (lat, lon), e.g. with rasterio:
    #   with rasterio.open("data/rasters/dem.tif") as src:
    #       feats["elevation"] = next(src.sample([(lon, lat)]))[0]
    return feats, {"source": "nearby_wells", "n_neighbors": int(len(idx)),
                   "nearest_km": round(float(dist.min()), 1)}


def get_features_detailed(lat: float, lon: float):
    """(feature_dict, info). Falls back to medians when no wells are near the point."""
    wells = load_wells()
    if wells is not None:
        feats, info = features_from_wells(wells, lat, lon)
        if feats is not None:
            art = joblib.load(C.MODEL_PATH)
            fill = {**art["feature_medians"], **art["feature_modes"]}
            return {**fill, **feats}, info
    else:
        info = {"source": "placeholder", "n_neighbors": 0, "nearest_km": None}
    art = joblib.load(C.MODEL_PATH)
    return {**art["feature_medians"], **art["feature_modes"]}, info


def get_features(lat: float, lon: float):
    """Backward-compatible: (feature_dict, is_real). is_real is False until you use rasters."""
    feats, _ = get_features_detailed(lat, lon)
    return feats, False
