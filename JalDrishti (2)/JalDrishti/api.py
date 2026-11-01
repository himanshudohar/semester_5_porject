"""JalDrishti - REST API.  Run:  uvicorn api:app --reload   then open http://127.0.0.1:8000/docs"""
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config as C
from geo_features import geocode, get_features_detailed
from predict import predict

app = FastAPI(title="JalDrishti API", version="1.0")


class Query(BaseModel):
    address: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


STATIC_DIR = C.ROOT / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def website():
    """The website: search box + map + result."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"name": "JalDrishti", "model_ready": C.MODEL_PATH.exists()}


@app.post("/predict")
def predict_location(q: Query):
    if not C.MODEL_PATH.exists():
        raise HTTPException(503, "Model not trained. Run python train.py")
    lat, lon, addr = q.lat, q.lon, None
    if lat is None or lon is None:
        if not q.address:
            raise HTTPException(400, "Provide address or lat+lon")
        g = geocode(q.address)
        if not g:
            raise HTTPException(404, "Address not found")
        lat, lon, addr = g
    feats, info = get_features_detailed(lat, lon)
    out = predict(feats)
    out.update({"lat": lat, "lon": lon, "address": addr,
                "features_are_real": False,            # True only once you use real rasters
                "feature_source": info["source"],      # "nearby_wells" or "placeholder"
                "n_neighbors": info["n_neighbors"],
                "nearest_km": info["nearest_km"]})
    return out
