"""
Unit tests for API endpoints.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from backend.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "data_loaded" in data


def test_list_wells(client):
    response = client.get("/wells")
    assert response.status_code == 200
    data = response.json()
    assert "wells" in data
    assert "count" in data


def test_map_data(client):
    response = client.get("/map/data")
    assert response.status_code == 200
    data = response.json()
    assert "wells" in data
    assert "faults" in data


def test_graph_structure(client):
    response = client.get("/graph/structure")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data


def test_predict_production(client):
    response = client.post("/predict/production", json={"well_id": "W0001", "horizon": 7})
    assert response.status_code in [200, 404]


def test_predict_spatial_impact(client):
    response = client.post("/predict/spatial-impact", json={"well_id": "W0001"})
    assert response.status_code in [200, 404]


def test_well_not_found(client):
    response = client.get("/well/NONEXISTENT")
    assert response.status_code == 404


def test_zones(client):
    response = client.get("/zones")
    assert response.status_code == 200
    data = response.json()
    assert "zones" in data
