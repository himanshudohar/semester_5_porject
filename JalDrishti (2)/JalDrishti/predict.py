"""Prediction + explanation helpers shared by the app and the API."""
import joblib
import pandas as pd
import config as C

_ART = None


def _art():
    global _ART
    if _ART is None:
        _ART = joblib.load(C.MODEL_PATH)
    return _ART


def category(p: float) -> str:
    if p < C.LOW_MAX:
        return "Low"
    if p < C.HIGH_MIN:
        return "Medium"
    return "High"


def predict(features: dict) -> dict:
    art = _art()
    row = pd.DataFrame([{k: features.get(k) for k in art["features"]}])
    p = float(art["model"].predict_proba(row)[:, 1][0])
    top = [r for r in art["importance"][:5]]
    return {
        "probability": round(p, 3),
        "potential": category(p),
        "top_factors": [{"feature": r["feature"], "importance": round(r["importance"], 4),
                         "value": features.get(r["feature"])} for r in top],
        "model": art["best_model"],
        "trained_on_demo_data": art["trained_on_demo_data"],
        "disclaimer": ("Groundwater potential estimate, not a guarantee. Consult a qualified "
                       "hydrogeologist / geophysical survey before drilling."),
    }
