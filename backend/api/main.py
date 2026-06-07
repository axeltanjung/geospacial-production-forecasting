"""
FastAPI Backend for Geospatial Oil & Gas Production Forecasting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from backend.services.forecast import ForecastService
from backend.utils.config import get_logger

logger = get_logger("api")

app = FastAPI(
    title="Geospatial Production Forecasting API",
    description="Spatio-Temporal AI System for Oil & Gas Reservoir Analytics",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = ForecastService()


class PredictionRequest(BaseModel):
    well_id: str
    horizon: int = 30


class SpatialImpactRequest(BaseModel):
    well_id: str


class ExplainRequest(BaseModel):
    well_id: str


@app.on_event("startup")
async def startup():
    logger.info("Starting Geospatial Production Forecasting API...")
    service.load()
    logger.info("API ready")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "geospatial-production-forecasting",
        "data_loaded": service.is_loaded,
        "num_wells": len(service.wells) if service.wells is not None else 0
    }


@app.post("/predict/production")
async def predict_production(request: PredictionRequest):
    result = service.predict_production(request.well_id, request.horizon)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/predict/spatial-impact")
async def predict_spatial_impact(request: SpatialImpactRequest):
    result = service.predict_spatial_impact(request.well_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/well/{well_id}")
async def get_well(well_id: str):
    result = service.get_well_data(well_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Well not found")
    return result


@app.get("/map/data")
async def get_map_data():
    return service.get_map_data()


@app.get("/graph/structure")
async def get_graph_structure():
    return service.get_graph_structure()


@app.post("/forecast/multi-step")
async def multi_step_forecast(request: PredictionRequest):
    return service.multi_step_forecast(request.well_id)


@app.post("/explain/spatial")
async def explain_spatial(request: ExplainRequest):
    result = service.predict_spatial_impact(request.well_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/wells")
async def list_wells():
    if service.wells is None:
        return {"wells": []}
    wells = service.wells[["well_id", "latitude", "longitude", "reservoir_zone", "depth"]].to_dict(orient="records")
    return {"wells": wells, "count": len(wells)}


@app.get("/zones")
async def list_zones():
    if service.wells is None:
        return {"zones": []}
    zones = service.wells.groupby("reservoir_zone").agg(
        num_wells=("well_id", "count"),
        avg_depth=("depth", "mean"),
        avg_permeability=("permeability_index", "mean")
    ).reset_index().to_dict(orient="records")
    return {"zones": zones}


if __name__ == "__main__":
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
