"""
PHOENIX Backend API — FastAPI skeleton
=========================================
Exposes the trained fire-risk model and the shelter/resource database
as a proper API, so any client (mobile app, SMS gateway, another
dashboard) can query it — not just the Streamlit dashboard.

Run locally:
    pip install -r requirements.txt
    uvicorn api:app --reload

Then open http://127.0.0.1:8000/docs for interactive API docs (Swagger UI).

Endpoints:
    GET  /health                     -> service status check
    POST /predict                    -> fire risk for one point/day (custom weather input)
    GET  /risk-map?date=YYYY-MM-DD   -> risk level for all grid cells on a given date
    GET  /shelters                   -> list shelters/resources (filterable by category)
    GET  /shelters/nearest           -> nearest available shelter to a given point
    GET  /alerts?date=YYYY-MM-DD     -> high-risk zones matched to nearest shelter (for SMS/USSD)
"""

from datetime import date as date_type
from math import radians, sin, cos, sqrt, atan2
from typing import Optional, List

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from phoenix_predict import predict_fire_risk
from risk_engine import get_alert
from air_quality import fetch_live_pm25

# -----------------------------------------------------------------
# App & data loading (runs once on startup)
# -----------------------------------------------------------------
app = FastAPI(
    title="PHOENIX API",
    description="Wildfire early-warning & shelter-matching API for North-East Algeria",
    version="1.0.0",
)

CLIMATE_DF = pd.read_csv("phoenix_climate_engineered.csv", parse_dates=["date"])
TODAY = pd.Timestamp.now().normalize()

SHELTERS_DF = pd.read_csv("north_algeria_shelters.csv")
SHELTERS_DF = SHELTERS_DF.drop(columns=["capacity", "name"])  # drop raw OSM tags (mostly empty / cause name clash)
SHELTERS_DF = SHELTERS_DF.rename(columns={"capacity_estimate": "capacity", "display_name": "name"})
SHELTERS_DF["name"] = SHELTERS_DF["name"].fillna(SHELTERS_DF["category"] + " (unnamed)")
SHELTERS_DF["capacity"] = pd.to_numeric(SHELTERS_DF["capacity"], errors="coerce").fillna(25).astype(int)
SHELTERS_DF["capacity_source"] = SHELTERS_DF["capacity_source"].fillna("unknown")
SHELTERS_DF["is_shelter"] = SHELTERS_DF["capacity_source"] != "default_estimate_non_shelter"
SHELTERS_DF["available"] = SHELTERS_DF["capacity"]  # no live occupancy feed yet

# Merge REAL current air-quality readings (Open-Meteo, per shelter location).
# This is a live snapshot, not tied to any specific historical date.
try:
    AQI_DF = pd.read_csv("shelters_with_air_quality.csv")[["osm_id", "pm2_5", "us_aqi", "observation_time"]]
    SHELTERS_DF = SHELTERS_DF.merge(AQI_DF, on="osm_id", how="left")
except FileNotFoundError:
    SHELTERS_DF["pm2_5"] = None
    SHELTERS_DF["us_aqi"] = None
    SHELTERS_DF["observation_time"] = None


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# -----------------------------------------------------------------
# Request / response schemas
# -----------------------------------------------------------------
class PredictRequest(BaseModel):
    lat: float = Field(..., json_schema_extra={"example": 36.7})
    lon: float = Field(..., json_schema_extra={"example": 4.5})
    doy: int = Field(..., ge=1, le=366, description="Day of year (1-366)")
    t2m_max: float = Field(..., description="Max daily temperature (°C)")
    t2m_min: float = Field(..., description="Min daily temperature (°C)")
    rh2m: float = Field(..., description="Relative humidity (%)")
    ws2m: float = Field(..., description="Wind speed (m/s)")
    prectotcorr: float = Field(..., description="Precipitation (mm)")


class PredictResponse(BaseModel):
    fire_probability: float
    risk_level: str


class ShelterOut(BaseModel):
    osm_id: int
    category: str
    name: str
    lat: float
    lon: float
    capacity: int
    available: int
    is_shelter: bool
    pm2_5: Optional[float] = None
    us_aqi: Optional[float] = None
    observation_time: Optional[str] = None


class NearestShelterResponse(BaseModel):
    name: str
    lat: float
    lon: float
    distance_km: float
    capacity: int
    available: int


class AlertOut(BaseModel):
    lat: float
    lon: float
    fire_probability: float
    risk_level: str
    nearest_shelter: Optional[str]
    nearest_shelter_distance_km: Optional[float]
    message: str


# -----------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "service": "PHOENIX API", "version": "1.0.0"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Run the model on custom weather input for a single point/day."""
    result = predict_fire_risk(
        lat=req.lat, lon=req.lon, doy=req.doy,
        t2m_max=req.t2m_max, t2m_min=req.t2m_min,
        rh2m=req.rh2m, ws2m=req.ws2m, prectotcorr=req.prectotcorr,
    )
    return result


@app.get("/risk-map", response_model=List[dict])
def risk_map(date: date_type = Query(..., description="Date to evaluate, e.g. 2026-08-12")):
    """Return fire risk for every grid cell in the region on a given date.
    Health/air-quality fields are only populated for TODAY (live data source);
    historical dates return null health fields with a note."""
    day_data = CLIMATE_DF[CLIMATE_DF["date"] == pd.Timestamp(date)]
    if day_data.empty:
        raise HTTPException(status_code=404, detail="No climate data available for this date.")

    is_today = pd.Timestamp(date).normalize() == TODAY
    doy = pd.Timestamp(date).dayofyear
    results = []
    for _, row in day_data.iterrows():
        r = predict_fire_risk(
            lat=row["LAT"], lon=row["LON"], doy=doy,
            t2m_max=row["T2M_MAX"], t2m_min=row["T2M_MIN"],
            rh2m=row["RH2M"], ws2m=row["WS2M"], prectotcorr=row["PRECTOTCORR"],
        )
        entry = {"lat": row["LAT"], "lon": row["LON"], **r,
                  "pm2_5": None, "health_level": None, "health_advice": None}
        if is_today:
            pm25 = fetch_live_pm25(row["LAT"], row["LON"])
            if pm25 is not None:
                alert = get_alert(r["fire_probability"], pm25)
                entry.update({"pm2_5": pm25, "health_level": alert.health_level,
                               "health_advice": alert.health_advice})
        results.append(entry)
    return results


@app.get("/shelters", response_model=List[ShelterOut])
def list_shelters(
    category: Optional[str] = Query(None, description="Filter: emergency_shelter, school, place_of_worship, fire_station, health_facility"),
    only_shelters: bool = Query(False, description="If true, exclude support-only resources (hospitals/fire stations)"),
):
    df = SHELTERS_DF
    if category:
        df = df[df["category"] == category]
    if only_shelters:
        df = df[df["is_shelter"]]
    cols = ["osm_id", "category", "name", "lat", "lon", "capacity", "available", "is_shelter",
            "pm2_5", "us_aqi", "observation_time"]
    return df[cols].to_dict(orient="records")


@app.get("/shelters/nearest", response_model=NearestShelterResponse)
def nearest_shelter(
    lat: float = Query(..., examples=[36.7]),
    lon: float = Query(..., examples=[4.5]),
    only_shelters: bool = Query(True, description="Restrict to actual evacuee shelters (exclude hospitals/fire stations)"),
):
    df = SHELTERS_DF[SHELTERS_DF["available"] > 0].copy()
    if only_shelters:
        df = df[df["is_shelter"]]
    if df.empty:
        raise HTTPException(status_code=404, detail="No available shelters found.")

    df["distance_km"] = df.apply(lambda s: haversine_km(lat, lon, s["lat"], s["lon"]), axis=1)
    nearest = df.loc[df["distance_km"].idxmin()]
    return {
        "name": nearest["name"], "lat": nearest["lat"], "lon": nearest["lon"],
        "distance_km": round(nearest["distance_km"], 2),
        "capacity": int(nearest["capacity"]), "available": int(nearest["available"]),
    }


@app.get("/alerts", response_model=List[AlertOut])
def alerts(date: date_type = Query(..., description="Date to evaluate, e.g. 2026-08-12")):
    """High-risk zones for a date, each matched to its nearest available shelter.
    This is the endpoint an SMS/USSD gateway would poll to send real alerts."""
    zones = risk_map(date)
    high_risk = [z for z in zones if z["risk_level"] == "High"]

    shelters = SHELTERS_DF[(SHELTERS_DF["is_shelter"]) & (SHELTERS_DF["available"] > 0)].copy()

    out = []
    for z in high_risk:
        nearest_name, nearest_dist = None, None
        if not shelters.empty:
            shelters["distance_km"] = shelters.apply(
                lambda s: haversine_km(z["lat"], z["lon"], s["lat"], s["lon"]), axis=1
            )
            nearest = shelters.loc[shelters["distance_km"].idxmin()]
            nearest_name, nearest_dist = nearest["name"], round(nearest["distance_km"], 2)

        health_note = ""
        if z.get("health_advice"):
            health_note = f" {z['health_advice']}"

        out.append({
            "lat": z["lat"], "lon": z["lon"],
            "fire_probability": z["fire_probability"], "risk_level": z["risk_level"],
            "nearest_shelter": nearest_name, "nearest_shelter_distance_km": nearest_dist,
            "message": (
                f"[ALERT] High wildfire risk near ({z['lat']}, {z['lon']}). "
                f"Probability: {z['fire_probability']*100:.0f}%. "
                + (f"Nearest shelter: {nearest_name} ({nearest_dist} km)." if nearest_name else "No nearby shelter capacity found.")
                + health_note
            ),
        })
    return out
