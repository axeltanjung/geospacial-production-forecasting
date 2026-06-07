"""
Unit tests for data generation and geospatial engine.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


def test_data_generator_initialization():
    from backend.simulation.data_generator import ReservoirFieldGenerator
    gen = ReservoirFieldGenerator(seed=42)
    assert gen.num_wells == 150
    assert gen.num_timesteps == 1460
    assert gen.num_zones == 5


def test_fault_generation():
    from backend.simulation.data_generator import ReservoirFieldGenerator
    gen = ReservoirFieldGenerator(seed=42)
    faults = gen.generate_fault_lines()
    assert len(faults) == 8
    assert all("start_lat" in f for f in faults)
    assert all(0 <= f["transmissibility"] <= 1 for f in faults)


def test_well_generation():
    from backend.simulation.data_generator import ReservoirFieldGenerator
    gen = ReservoirFieldGenerator(seed=42)
    gen.generate_fault_lines()
    wells = gen.generate_wells()
    assert len(wells) == 150
    assert "latitude" in wells.columns
    assert "longitude" in wells.columns
    assert "reservoir_zone" in wells.columns
    assert wells["permeability_index"].min() > 0


def test_geospatial_engine():
    from backend.simulation.data_generator import ReservoirFieldGenerator
    from backend.geospatial.engine import GeospatialEngine

    gen = ReservoirFieldGenerator(seed=42)
    gen.generate_fault_lines()
    wells = gen.generate_wells()

    engine = GeospatialEngine(wells)
    dist_matrix = engine.compute_distance_matrix()
    assert dist_matrix.shape == (150, 150)
    assert np.all(np.diag(dist_matrix) == 0)

    adj = engine.build_adjacency_graph()
    assert adj.shape == (150, 150)
    assert np.all(np.diag(adj) == 0)


def test_graph_builder():
    from backend.simulation.data_generator import ReservoirFieldGenerator
    from backend.graph.builder import WellGraphBuilder

    gen = ReservoirFieldGenerator(seed=42)
    gen.generate_fault_lines()
    wells = gen.generate_wells()

    builder = WellGraphBuilder(wells, gen.distance_matrix)
    graph = builder.build_knn_graph(k=5)
    assert graph.number_of_nodes() == 150
    assert graph.number_of_edges() > 0

    stats = builder.get_graph_stats(graph)
    assert stats["avg_degree"] > 0


def test_point_to_line_distance():
    from backend.simulation.data_generator import ReservoirFieldGenerator
    dist = ReservoirFieldGenerator._point_to_line_distance(0, 0, 1, 0, 1, 1)
    assert dist > 0

    dist2 = ReservoirFieldGenerator._point_to_line_distance(1, 0.5, 1, 0, 1, 1)
    assert dist2 < 0.01
