"""
PHOENIX — Wildfire Early Warning & Shelter Matching Dashboard
================================================================
Run locally:
    pip install streamlit folium streamlit-folium xgboost pandas
    streamlit run streamlit_app.py

Or deploy free on Streamlit Community Cloud (streamlit.io/cloud):
    Push this file + phoenix_xgb_final.json + phoenix_predict.py +
    phoenix_climate_engineered.csv + shelters.csv to a GitHub repo,
    then connect the repo on share.streamlit.io.
"""

import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2

from phoenix_predict import predict_fire_risk

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="PHOENIX — Fire Early Warning", layout="wide")
st.title("🔥 PHOENIX — Wildfire Early Warning & Shelter Matching")
st.caption("North-East Algeria (Kabylie / Annaba region) — AI for All Hackathon")

COLOR_MAP = {"Low": "green", "Medium": "orange", "High": "red"}

# ---------------------------------------------------------------
# Load data (cached so it only loads once per session)
# ---------------------------------------------------------------
@st.cache_data
def load_climate():
    df = pd.read_csv("phoenix_climate_engineered.csv", parse_dates=["date"])
    return df

@st.cache_data
def load_shelters():
    # Real OSM data: schools, mosques, health facilities, fire stations,
    # and declared emergency shelters across North-East Algeria.
    df = pd.read_csv("north_algeria_shelters.csv")
    df = df.drop(columns=["capacity", "name"])  # drop raw OSM tags (mostly empty / cause name clash)
    df = df.rename(columns={"capacity_estimate": "capacity", "display_name": "name"})
    # Defensive cleanup: guarantee no NaN ever reaches the map/popups,
    # regardless of upstream data quality.
    df["name"] = df["name"].fillna(df["category"] + " (unnamed)")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(25).astype(int)
    # Health facilities & fire stations are SUPPORT resources, not evacuee
    # shelters (their capacity_source is "default_estimate_non_shelter") —
    # they should not be counted as usable shelter capacity for evacuees.
    df["capacity_source"] = df["capacity_source"].fillna("unknown")
    df["is_shelter"] = df["capacity_source"] != "default_estimate_non_shelter"
    # "available" isn't tracked live yet (no real-time feed) -> assume full capacity available
    # until a real occupancy-reporting mechanism (e.g. volunteer/NGO updates) is connected.
    df["available"] = df["capacity"]
    return df[["osm_id", "category", "name", "lat", "lon", "capacity", "available",
               "capacity_source", "is_shelter"]]

climate = load_climate()
shelters_all = load_shelters()

# ---------------------------------------------------------------
# Sidebar: choose which facility types to show
# ---------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Locations to show on map")
CATEGORY_LABELS = {
    "emergency_shelter": "🏠 Declared emergency shelters",
    "school": "🏫 Schools (evacuee shelters)",
    "place_of_worship": "🕌 Places of worship (evacuee shelters)",
    "fire_station": "🚒 Civil protection stations (support only)",
    "health_facility": "🏥 Health facilities (support only)",
}
selected_categories = [
    cat for cat, label in CATEGORY_LABELS.items()
    if st.sidebar.checkbox(label, value=(cat in ["emergency_shelter", "school", "place_of_worship"]))
]
shelters = shelters_all[shelters_all["category"].isin(selected_categories)].copy()
st.sidebar.caption(f"{len(shelters)} locations match filters out of {len(shelters_all)} total.")
st.sidebar.caption("Note: fire stations & health facilities are support resources, "
                    "not counted as evacuee shelter capacity.")
max_markers = st.sidebar.slider("Max shelter markers drawn on map (performance)", 50, 2000, 300, step=50)
st.sidebar.caption("Lower this if the map feels slow to load. Largest-capacity locations are shown first; "
                    "matching/metrics below still use ALL selected locations, not just the ones drawn.")

# ---------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------
st.sidebar.header("Controls")
available_dates = sorted(climate["date"].unique())
selected_date = st.sidebar.select_slider(
    "Select date",
    options=available_dates,
    value=available_dates[-1],
    format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Risk thresholds**\n\n"
    "- 🟢 Low: probability < 0.35\n"
    "- 🟡 Medium: 0.35 – 0.65\n"
    "- 🔴 High: probability ≥ 0.65"
)

# ---------------------------------------------------------------
# Run model on all grid cells for the selected date
# ---------------------------------------------------------------
day_data = climate[climate["date"] == selected_date].copy()
doy = pd.Timestamp(selected_date).dayofyear

results = []
for _, row in day_data.iterrows():
    r = predict_fire_risk(
        lat=row["LAT"], lon=row["LON"], doy=doy,
        t2m_max=row["T2M_MAX"], t2m_min=row["T2M_MIN"],
        rh2m=row["RH2M"], ws2m=row["WS2M"], prectotcorr=row["PRECTOTCORR"],
    )
    results.append({**row.to_dict(), **r})
res_df = pd.DataFrame(results)

# ---------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("🔴 High risk zones", int((res_df["risk_level"] == "High").sum()))
col2.metric("🟡 Medium risk zones", int((res_df["risk_level"] == "Medium").sum()))
col3.metric("🟢 Low risk zones", int((res_df["risk_level"] == "Low").sum()))
shelter_only = shelters[shelters["is_shelter"]]
col4.metric("Shelters with capacity", int((shelter_only["available"] > 0).sum()))

# ---------------------------------------------------------------
# Map
# ---------------------------------------------------------------
st.subheader("Live Risk Map")
center_lat, center_lon = res_df["LAT"].mean(), res_df["LON"].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="CartoDB positron")

for _, row in res_df.iterrows():
    folium.CircleMarker(
        location=[row["LAT"], row["LON"]],
        radius=16,
        color=COLOR_MAP[row["risk_level"]],
        fill=True,
        fill_color=COLOR_MAP[row["risk_level"]],
        fill_opacity=0.7,
        weight=2,
        popup=folium.Popup(
            f"<b>{row['risk_level']} risk</b><br>"
            f"Fire probability: {row['fire_probability']*100:.1f}%<br>"
            f"Max Temp: {row['T2M_MAX']:.1f}°C | Humidity: {row['RH2M']:.1f}%",
            max_width=220,
        ),
        tooltip=f"{row['risk_level']} risk",
    ).add_to(m)

shelter_cluster = MarkerCluster(name="Shelters & Resources").add_to(m)
shelters_to_draw = shelters.sort_values("capacity", ascending=False).head(max_markers)
for _, s in shelters_to_draw.iterrows():
    if s["is_shelter"]:
        color, kind = "blue", "Shelter"
    else:
        color, kind = "darkcyan", "Support resource (not for housing evacuees)"
    folium.CircleMarker(
        location=[s["lat"], s["lon"]],
        radius=7,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        weight=1,
        popup=folium.Popup(
            f"<b>{s['name']}</b><br>{kind}<br>Est. capacity: {s['available']}/{s['capacity']}",
            max_width=220,
        ),
        tooltip=s["name"],
    ).add_to(shelter_cluster)

if len(shelters) > max_markers:
    st.caption(f"Showing the {max_markers} largest-capacity locations out of {len(shelters)} matching your filters "
               f"(adjust the slider in the sidebar to show more). All {len(shelters)} are still used in the "
               f"shelter-matching table below.")

st_folium(m, width=1100, height=550, returned_objects=[])

# ---------------------------------------------------------------
# High-risk zones -> nearest shelter matching
# ---------------------------------------------------------------
def haversine_km_vec(lat1, lon1, lat_arr, lon_arr):
    """Vectorized haversine distance (km) from one point to an array of points."""
    import numpy as np
    R = 6371
    lat1r, lon1r = radians(lat1), radians(lon1)
    lat2r, lon2r = np.radians(lat_arr), np.radians(lon_arr)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

st.subheader("⚠️ High-Risk Zones → Nearest Available Shelter")
high_risk = res_df[res_df["risk_level"] == "High"].copy()

if high_risk.empty:
    st.success("No high-risk zones detected for this date.")
else:
    matches = []
    avail = shelters[(shelters["is_shelter"]) & (shelters["available"] > 0)].copy()
    for _, zone in high_risk.iterrows():
        if avail.empty:
            matches.append({"Zone (lat,lon)": f"{zone['LAT']}, {zone['LON']}",
                             "Fire probability": f"{zone['fire_probability']*100:.1f}%",
                             "Nearest shelter": "⚠️ No capacity available nearby"})
            continue
        dists = haversine_km_vec(zone["LAT"], zone["LON"], avail["lat"].values, avail["lon"].values)
        nearest_idx = dists.argmin()
        nearest = avail.iloc[nearest_idx]
        matches.append({
            "Zone (lat,lon)": f"{zone['LAT']}, {zone['LON']}",
            "Fire probability": f"{zone['fire_probability']*100:.1f}%",
            "Nearest shelter": nearest["name"],
            "Distance (km)": f"{dists[nearest_idx]:.1f}",
            "Shelter capacity": f"{nearest['available']}/{nearest['capacity']}",
        })
    st.dataframe(pd.DataFrame(matches), use_container_width=True)

# ---------------------------------------------------------------
# Simulated SMS alert log
# ---------------------------------------------------------------
st.subheader("📱 Simulated SMS/USSD Alerts")
if not high_risk.empty:
    for _, zone in high_risk.iterrows():
        st.code(
            f"[ALERT] High wildfire risk near ({zone['LAT']}, {zone['LON']}). "
            f"Probability: {zone['fire_probability']*100:.0f}%. "
            f"Move livestock/valuables and check nearest shelter now.",
            language=None,
        )
else:
    st.info("No alerts to send for this date.")

st.caption("Data sources: NASA FIRMS (fire history) · NASA POWER (climate) · Model: XGBoost, threshold-tuned")
