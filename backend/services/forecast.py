"""
Forecast Service: Business logic for production forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import json

from backend.utils.config import DATA_DIR, MODELS_DIR, get_logger

logger = get_logger("forecast_service")


class ForecastService:
    def __init__(self):
        self.wells = None
        self.temporal = None
        self.faults = None
        self.baseline_model = None
        self.scaler = None
        self.is_loaded = False

    def load(self):
        try:
            self.wells = pd.read_csv(DATA_DIR / "wells_static.csv")
            self.temporal = pd.read_csv(DATA_DIR / "production_temporal.csv")
            self.temporal["timestamp"] = pd.to_datetime(self.temporal["timestamp"])

            with open(DATA_DIR / "fault_lines.json") as f:
                self.faults = json.load(f)

            if (MODELS_DIR / "xgboost_model.joblib").exists():
                import joblib
                self.baseline_model = joblib.load(MODELS_DIR / "xgboost_model.joblib")
                self.scaler = joblib.load(MODELS_DIR / "scaler.joblib")

            self.is_loaded = True
            logger.info("ForecastService loaded successfully")
        except Exception as e:
            logger.warning(f"ForecastService partial load: {e}")
            self.is_loaded = self.wells is not None

    def get_well_data(self, well_id: str) -> Optional[Dict]:
        if self.wells is None:
            return None
        well = self.wells[self.wells["well_id"] == well_id]
        if well.empty:
            return None
        well_info = well.iloc[0].to_dict()
        history = self.temporal[self.temporal["well_id"] == well_id].tail(365)
        well_info["production_history"] = history[
            ["timestamp", "production_rate",
             "reservoir_pressure", "water_cut"]
        ].to_dict(orient="records")
        return well_info

    def get_map_data(self) -> Dict:
        if self.wells is None:
            return {"wells": [], "faults": []}

        latest = self.temporal.groupby("well_id").last().reset_index()
        merged = self.wells.merge(
            latest[["well_id", "production_rate", "reservoir_pressure",
                    "water_cut"]], on="well_id", how="left")

        wells_data = []
        for _, row in merged.iterrows():
            wells_data.append({
                "well_id": row["well_id"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "production_rate": float(row.get("production_rate", 0) or 0),
                "reservoir_pressure": float(row.get("reservoir_pressure", 0) or 0),
                "water_cut": float(row.get("water_cut", 0) or 0),
                "depth": float(row["depth"]),
                "zone": row["reservoir_zone"],
                "permeability": float(row["permeability_index"])
            })

        return {
            "wells": wells_data,
            "faults": self.faults or [],
            "field_bounds": {
                "lat_min": float(self.wells["latitude"].min()),
                "lat_max": float(self.wells["latitude"].max()),
                "lon_min": float(self.wells["longitude"].min()),
                "lon_max": float(self.wells["longitude"].max())
            }
        }

    def predict_production(self, well_id: str, horizon: int = 30) -> Dict:
        well_data = self.temporal[self.temporal["well_id"] == well_id].sort_values("timestamp")
        if well_data.empty:
            return {"error": "Well not found"}

        recent = well_data.tail(90)
        current_rate = float(recent["production_rate"].iloc[-1])
        avg_decline = float(recent["production_rate"].pct_change().mean())

        forecast = []
        rate = current_rate
        for day in range(1, horizon + 1):
            noise = np.random.normal(0, current_rate * 0.02)
            rate = rate * (1 + avg_decline) + noise
            rate = max(rate, 10)
            uncertainty = current_rate * 0.05 * np.sqrt(day)
            forecast.append({
                "day": day,
                "predicted_rate": round(rate, 2),
                "lower_bound": round(max(rate - 1.96 * uncertainty, 0), 2),
                "upper_bound": round(rate + 1.96 * uncertainty, 2)
            })

        return {
            "well_id": well_id,
            "current_rate": current_rate,
            "forecast_horizon": horizon,
            "forecast": forecast,
            "avg_decline_rate": round(avg_decline * 100, 3)
        }

    def predict_spatial_impact(self, well_id: str) -> Dict:
        well = self.wells[self.wells["well_id"] == well_id]
        if well.empty:
            return {"error": "Well not found"}

        well_info = well.iloc[0]
        coords = self.wells[["latitude", "longitude"]].values
        well_coord = np.array([[well_info["latitude"], well_info["longitude"]]])

        from scipy.spatial.distance import cdist
        distances = cdist(well_coord, coords)[0]

        neighbors = []
        for i in np.argsort(distances)[1:11]:
            neighbor = self.wells.iloc[i]
            n_data = self.temporal[self.temporal["well_id"] == neighbor["well_id"]].tail(1)
            n_prod = float(n_data["production_rate"].iloc[0]) if not n_data.empty else 0

            connectivity = float(np.exp(-distances[i] / 0.05))
            impact = connectivity * n_prod * 0.01

            neighbors.append({
                "well_id": neighbor["well_id"],
                "distance": round(float(distances[i]), 6),
                "connectivity": round(connectivity, 4),
                "production_rate": round(n_prod, 2),
                "estimated_impact": round(impact, 2),
                "zone": neighbor["reservoir_zone"],
                "same_zone": bool(well_info["reservoir_zone"] == neighbor["reservoir_zone"])
            })

        return {
            "well_id": well_id,
            "zone": well_info["reservoir_zone"],
            "neighbors": neighbors,
            "total_spatial_impact": round(sum(n["estimated_impact"] for n in neighbors), 2)
        }

    def get_graph_structure(self) -> Dict:
        if self.wells is None:
            return {"nodes": [], "edges": []}

        from backend.geospatial.engine import GeospatialEngine
        engine = GeospatialEngine(self.wells)
        engine.compute_distance_matrix()
        engine.build_adjacency_graph(self.faults)

        nodes = []
        for _, row in self.wells.iterrows():
            nodes.append({
                "id": row["well_id"],
                "lat": row["latitude"],
                "lon": row["longitude"],
                "zone": row["reservoir_zone"]
            })

        edges = engine.get_edge_list()
        return {"nodes": nodes, "edges": edges[:500]}

    def multi_step_forecast(self, well_id: str) -> Dict:
        results = {}
        for horizon in [1, 7, 30]:
            results[f"t+{horizon}"] = self.predict_production(well_id, horizon=horizon)
        return {"well_id": well_id, "forecasts": results}
