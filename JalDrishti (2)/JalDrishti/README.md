# JalDrishti — AI-Based Groundwater Potential Prediction System

*"Check the groundwater potential before you drill."*

A decision-support tool that estimates groundwater potential (Low / Medium / High) for a location.
**It is a risk estimate, not a guarantee. Always get a hydrogeological/geophysical survey before drilling.**

> ⚠️ The included demo data is **synthetic**. The pipeline runs end to end, but predictions mean
> nothing until you train on real, documented well observations (Step 6).

## Project structure

```
JalDrishti/
├── config.py               # features, paths, thresholds
├── generate_demo_data.py   # makes SYNTHETIC data so everything runs
├── train.py                # baselines, spatial CV, calibration, importance, save model
├── geo_features.py         # geocoding + feature lookup (replace placeholder with real data)
├── predict.py              # prediction + Low/Medium/High + explanation
├── app.py                  # Streamlit map UI (quick internal version)
├── api.py                  # FastAPI: serves the website + REST API (/predict)
├── static/index.html       # The website (search box, map, result gauge)
├── requirements.txt
├── data/   models/   reports/
```

## Step-by-step in VS Code

### Step 1 — Install tools
1. Install **Python 3.10+** from python.org (tick *Add Python to PATH* on Windows).
2. Install **VS Code** from code.visualstudio.com.
3. In VS Code open Extensions (`Ctrl+Shift+X`) and install **Python** and **Jupyter** (by Microsoft).

### Step 2 — Open the project
1. Unzip `JalDrishti.zip`.
2. VS Code → **File → Open Folder…** → select the `JalDrishti` folder.
3. Open the terminal: **Terminal → New Terminal** (`` Ctrl+` ``).

### Step 3 — Create a virtual environment
```bash
python -m venv venv
```
Activate it:
- Windows (PowerShell): `venv\Scripts\Activate.ps1`
  (if blocked: `Set-ExecutionPolicy -Scope Process Bypass`)
- Mac/Linux: `source venv/bin/activate`

Then press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the one inside `venv`.

### Step 4 — Install libraries
```bash
pip install -r requirements.txt
```

### Step 5 — Run the demo pipeline
```bash
python generate_demo_data.py     # creates data/demo_wells.csv (synthetic)
python train.py                  # trains, validates by region, saves models/ + reports/
streamlit run app.py             # opens the map app at http://localhost:8501
```
**The website (recommended for users):**
```bash
uvicorn api:app --reload
```
Open **http://127.0.0.1:8000** in your browser. Type a village/address (or tap the map, or press
"Use my location") and the page shows Low / Medium / High potential, the probability on a borehole-style
gauge, the main factors, and the next steps. The website is `static/index.html`; the API docs are at `/docs`.
(Needs internet for the map tiles and address search.)

To let others on your Wi-Fi open it: `uvicorn api:app --host 0.0.0.0 --port 8000`, then visit
`http://<your-computer-ip>:8000` from a phone. To put it online, deploy this folder on Render, Railway or
a VPS with the start command `uvicorn api:app --host 0.0.0.0 --port $PORT`.

`train.py` prints precision, recall, F1, ROC-AUC, confusion-matrix counts and Brier score using
**geographic cross-validation** (whole 15 km blocks held out), and saves a feature-importance chart
in `reports/`.

### Step 6 — Replace demo data with REAL data (this is the actual project)
1. **Pick one study area** (e.g. Ujjain district).
2. **Collect well/borewell records**: coordinates, success/failure, depth, yield. Sources to explore:
   India-WRIS, CGWB (Central Ground Water Board) well data, state groundwater departments,
   local drillers/farmers (record carefully).
3. **Define the label rule in writing**, e.g. `label = 1` if yield ≥ X litres/min at ≤ Y m depth.
4. **Collect spatial layers** (free): SRTM DEM (elevation, derive slope), CHIRPS rainfall, Bhukosh/GSI
   geology, soil maps (NBSS&LUP / SoilGrids), ESA WorldCover land cover, Sentinel-2 NDVI, OSM/HydroSHEDS streams.
5. **Build `data/wells.csv`** with these columns (same names as `config.py`):
   `lat, lon, rainfall, elevation, slope, dist_stream, drainage_density, ndvi,
   nearby_well_success_rate, nearby_well_avg_depth, soil, geology, landcover, label`
   (extra post-drilling columns like `well_depth`, `water_level`, `yield_lpm` may exist but are never used as inputs — that is leakage).
   Use `geopandas` + `rasterio` to sample rasters at each well coordinate.
   For `nearby_well_*`, compute from *other* wells only, never the well itself.
6. Run `python train.py` again. It automatically prefers `data/wells.csv` over the demo file.
7. Aim for hundreds to thousands of wells. If spatial-CV ROC-AUC is near 0.5, the data/features are not informative — improve data before improving models.

### Step 7 — Make the app use real features
How the app gets inputs for a clicked point: `geo_features.py` finds the 8 nearest wells in
`data/wells.csv` (or the demo file) within 25 km. It uses their success rate and depth as
`nearby_well_*`, and a distance-weighted average of their other values for the remaining features.
That is why results change from place to place. The result page says "estimated from the N nearest wells".
- **No wells within 25 km** (outside your study area): the app shows a warning and a generic average.
  Predictions are only meaningful inside the area where you have well data.
- **Upgrade for the real project:** in `features_from_wells()` overwrite elevation, slope, rainfall,
  geology, soil, land cover, stream distance and NDVI with exact values sampled from your rasters/shapefiles
  at the point (comments show a `rasterio` example). Keep the same feature names.
- If you change the data, run `python generate_demo_data.py` (demo only) and `python train.py` again.

### Step 8 — Debug / run in VS Code
- Run a file: open it → ▶ button top-right.
- Debug: set breakpoint → `F5`.
- Explore data interactively: create `eda.ipynb`, select the `venv` kernel, and run
  `import pandas as pd; df = pd.read_csv("data/wells.csv"); df.describe()`.

### Step 9 — Deploy a prototype
Push to GitHub, then deploy `app.py` on **Streamlit Community Cloud** (free). Make sure
`models/jaldrishti_model.joblib` is committed or training runs at startup.

## Roadmap mapping
| Roadmap phase | Where |
|---|---|
| 1–3 target, data, feature table | Step 6, `config.py` |
| 4 EDA | notebook (Step 8) |
| 5–6 baselines + tuned models | `train.py` |
| 7 spatial validation | `train.py` (GroupKFold on lat/lon blocks) |
| 8 explainability + calibration | `train.py` (permutation importance, isotonic calibration) |
| 9 address→map interface | `geo_features.py`, `app.py` |
| 10–11 prototype + field validation | Step 9 + log real drilling outcomes |

## Honest limitations
- Accuracy depends entirely on real data quality; synthetic results are meaningless.
- Nearby-well features are computed from all wells, so geographic validation scores can be slightly optimistic.
  For a final report, compute them separately inside each validation fold.
- Feature importance shown is model-wide, not per-location SHAP (add `shap` later for local explanations).
- Nominatim geocoding is free but rate-limited (1 req/sec); check terms or use Google Geocoding.
