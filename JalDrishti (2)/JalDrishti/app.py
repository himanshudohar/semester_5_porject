"""JalDrishti - Streamlit UI.  Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

import config as C
from geo_features import geocode, get_features
from predict import predict

st.set_page_config(page_title="JalDrishti", page_icon="💧", layout="wide")
st.title("💧 JalDrishti — Groundwater Potential Prediction")
st.caption("Check the groundwater potential before you drill.")

if not C.MODEL_PATH.exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

if "loc" not in st.session_state:
    st.session_state.loc = (23.1765, 75.7885)  # Ujjain default
    st.session_state.addr = "Ujjain, Madhya Pradesh"

left, right = st.columns([1.2, 1])

with left:
    q = st.text_input("Search village / address")
    if st.button("Search") and q:
        res = geocode(q)
        if res:
            st.session_state.loc, st.session_state.addr = (res[0], res[1]), res[2]
        else:
            st.warning("Address not found. Click on the map or enter coordinates.")
    lat, lon = st.session_state.loc
    m = folium.Map(location=[lat, lon], zoom_start=12)
    folium.Marker([lat, lon], tooltip="Selected location").add_to(m)
    if C.REAL_DATA_PATH.exists():
        wells = pd.read_csv(C.REAL_DATA_PATH)
        for _, r in wells.head(500).iterrows():
            folium.CircleMarker([r.lat, r.lon], radius=3,
                                color="green" if r.label == 1 else "red", fill=True).add_to(m)
    out = st_folium(m, height=450, width=None)
    if out and out.get("last_clicked"):
        st.session_state.loc = (out["last_clicked"]["lat"], out["last_clicked"]["lng"])
        st.session_state.addr = "Map-selected point"
        st.rerun()

with right:
    lat, lon = st.session_state.loc
    st.subheader("Selected location")
    st.write(st.session_state.addr)
    st.write(f"**Latitude:** {lat:.5f}  \n**Longitude:** {lon:.5f}")
    feats, is_real = get_features(lat, lon)
    res = predict(feats)

    color = {"Low": "🔴", "Medium": "🟠", "High": "🟢"}[res["potential"]]
    st.metric("Groundwater potential", f"{color} {res['potential']}")
    st.progress(min(max(res["probability"], 0.0), 1.0),
                text=f"Calibrated probability: {res['probability']*100:.0f}%")

    if res["trained_on_demo_data"]:
        st.error("⚠️ Model trained on SYNTHETIC demo data. Results are NOT real.")
    if not is_real:
        st.warning("Environmental features are placeholder study-area medians "
                   "(connect real raster/shapefile data in geo_features.py).")

    st.subheader("Main factors (model-wide importance)")
    st.dataframe(pd.DataFrame(res["top_factors"]), hide_index=True)
    with st.expander("All input features"):
        st.json(feats)

    st.info(res["disclaimer"])
    st.markdown("**Next step:** get a local hydrogeological / geophysical (e.g. resistivity) survey before drilling.")
