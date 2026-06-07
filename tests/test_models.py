"""
Tests for ML models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest


def test_gcn_forward():
    from backend.models.gnn.model import WellGCN
    model = WellGCN(input_dim=10, hidden_dim=32, output_dim=3)
    x = torch.randn(20, 10)
    adj = torch.rand(20, 20) > 0.7
    adj = adj.float()
    out = model(x, adj)
    assert out.shape == (20, 3)


def test_gnn_uncertainty():
    from backend.models.gnn.model import WellProductionGNN
    model = WellProductionGNN(input_dim=10, hidden_dim=32, output_dim=3)
    x = torch.randn(20, 10)
    adj = (torch.rand(20, 20) > 0.7).float()
    mean, std = model(x, adj)
    assert mean.shape == (20, 3)
    assert std.shape == (20, 3)
    assert (std > 0).all()


def test_stgnn_forward():
    from backend.models.spatio_temporal.model import SpatioTemporalGNN
    model = SpatioTemporalGNN(
        temporal_input_dim=5,
        spatial_input_dim=4,
        output_dim=1
    )
    temporal = torch.randn(10, 30, 5)
    spatial = torch.randn(10, 4)
    adj = (torch.rand(10, 10) > 0.5).float()
    mean, std = model(temporal, spatial, adj)
    assert mean.shape == (10, 1)
    assert std.shape == (10, 1)


def test_gp_fallback():
    from backend.models.gaussian_process.model import SpatialGaussianProcess
    gp = SpatialGaussianProcess()
    coords = np.random.rand(30, 2)
    values = np.random.rand(30) * 100
    result = gp.fit(coords, values)
    assert gp.is_trained
    mean, lower, upper = gp.predict(coords[:5])
    assert len(mean) == 5
    assert np.all(lower <= upper)


def test_baseline_prepare():
    from backend.models.baseline_ml.model import BaselineModels
    import pandas as pd
    baseline = BaselineModels()
    temporal = pd.DataFrame({
        "well_id": ["W001"] * 10,
        "production_rate": np.random.rand(10) * 100,
        "water_cut": np.random.rand(10),
        "gas_oil_ratio": np.random.rand(10) * 1000,
        "reservoir_pressure": np.random.rand(10) * 5000,
        "injection_rate": np.random.rand(10) * 50,
        "neighbor_well_avg_production": np.random.rand(10) * 100,
        "neighbor_well_pressure_avg": np.random.rand(10) * 5000,
        "spatial_density_index": np.random.rand(10),
        "local_depletion_factor": np.random.rand(10),
        "decline_rate": np.random.rand(10) * 0.01,
        "production_momentum": np.random.rand(10) * 100,
        "pressure_diffusion_index": np.random.rand(10),
        "future_production_t1": np.random.rand(10) * 100,
    })
    wells = pd.DataFrame({
        "well_id": ["W001"],
        "depth": [8000],
        "permeability_index": [20],
        "porosity_index": [0.15],
        "distance_to_fault_line": [0.05],
        "distance_to_boundary": [0.1]
    })
    X, y = baseline.prepare_data(temporal, wells)
    assert X.shape[0] == 10
    assert len(y) == 10
