"""
PHOENIX Fire Risk Prediction — Final Model Package (v2 - stable format)
=========================================================================
Contains:
  - phoenix_xgb_final.json  -> trained XGBoost model (official save format,
                                 no version-mismatch warnings)
  - risk_level()             -> converts probability to Low/Medium/High
  - predict_fire_risk()      -> ready-to-use function for the API

Usage:
    from phoenix_predict import predict_fire_risk
    result = predict_fire_risk(lat=36.7, lon=4.5, doy=213,
                                t2m_max=34.5, t2m_min=21.0,
                                rh2m=28.0, ws2m=4.2, prectotcorr=0.0)
    print(result)
    # {'fire_probability': 0.87, 'risk_level': 'High'}
"""

from xgboost import XGBClassifier
import pandas as pd

MODEL_PATH = "phoenix_xgb_final.json"
THRESHOLD_LOW = 0.35
THRESHOLD_HIGH = 0.65

_model = XGBClassifier()
_model.load_model(MODEL_PATH)

def risk_level(p: float) -> str:
    """Convert a fire probability (0-1) into a Low/Medium/High risk label."""
    if p < THRESHOLD_LOW:
        return "Low"
    elif p < THRESHOLD_HIGH:
        return "Medium"
    else:
        return "High"

def predict_fire_risk(lat, lon, doy, t2m_max, t2m_min, rh2m, ws2m, prectotcorr):
    """
    Run the trained model on a single point's weather data and
    return both the raw probability and the risk category.
    """
    X = pd.DataFrame([{
        "LAT": lat, "LON": lon, "DOY": doy,
        "T2M_MAX": t2m_max, "T2M_MIN": t2m_min,
        "RH2M": rh2m, "WS2M": ws2m, "PRECTOTCORR": prectotcorr
    }])
    proba = float(_model.predict_proba(X)[0, 1])
    return {
        "fire_probability": round(proba, 4),
        "risk_level": risk_level(proba)
    }

if __name__ == "__main__":
    example = predict_fire_risk(
        lat=36.7, lon=4.5, doy=213,
        t2m_max=34.5, t2m_min=21.0,
        rh2m=28.0, ws2m=4.2, prectotcorr=0.0
    )
    print("Example prediction:", example)
