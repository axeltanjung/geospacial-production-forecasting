"""
Geospatial Engine: Core geospatial computations for well analysis.

Handles:
- Distance matrix computation
- Adjacency graph construction
- Spatial interpolation (kriging-style)
- Reservoir zone clustering
- Spatial feature engineering
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from scipy.spatial import Voronoi
import networkx as nx
from typing import Tuple, Dict, List, Optional


class GeospatialEngine:
    def __init__(self, wells_df: pd.DataFrame, connectivity_threshold: float = 0.05):
        self.wells = wells_df
        self.coords = wells_df[["latitude", "longitude"]].values
        self.threshold = connectivity_threshold
        self.distance_matrix = None
        self.adjacency_matrix = None
        self.graph = None

    def compute_distance_matrix(self) -> np.ndarray:
        self.distance_matrix = cdist(self.coords, self.coords, metric="euclidean")
        return self.distance_matrix

    def build_adjacency_graph(self, fault_lines: Optional[List[Dict]] = None) -> np.ndarray:
        if self.distance_matrix is None:
            self.compute_distance_matrix()

        adjacency = (self.distance_matrix < self.threshold).astype(float)
        np.fill_diagonal(adjacency, 0)

        weights = np.exp(-self.distance_matrix / (self.threshold / 2))
        self.adjacency_matrix = adjacency * weights

        if fault_lines:
            self._apply_fault_barriers(fault_lines)

        return self.adjacency_matrix

    def _apply_fault_barriers(self, fault_lines: List[Dict]):
        for i in range(len(self.coords)):
            for j in range(i + 1, len(self.coords)):
                if self.adjacency_matrix[i, j] == 0:
                    continue
                for fault in fault_lines:
                    if self._segments_intersect(
                        self.coords[i], self.coords[j],
                        np.array([fault["start_lat"], fault["start_lon"]]),
                        np.array([fault["end_lat"], fault["end_lon"]])
                    ):
                        factor = fault.get("transmissibility", 0.1)
                        self.adjacency_matrix[i, j] *= factor
                        self.adjacency_matrix[j, i] *= factor
                        break

    @staticmethod
    def _segments_intersect(p1, p2, p3, p4) -> bool:
        d1 = p2 - p1
        d2 = p4 - p3
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-10:
            return False
        t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
        u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
        return 0 <= t <= 1 and 0 <= u <= 1

    def build_networkx_graph(self) -> nx.Graph:
        if self.adjacency_matrix is None:
            self.build_adjacency_graph()

        self.graph = nx.Graph()
        well_ids = self.wells["well_id"].tolist()

        for i, wid in enumerate(well_ids):
            self.graph.add_node(wid, **{
                "latitude": self.coords[i, 0],
                "longitude": self.coords[i, 1],
                "zone": self.wells.iloc[i].get("reservoir_zone", "unknown")
            })

        for i in range(len(well_ids)):
            for j in range(i + 1, len(well_ids)):
                weight = self.adjacency_matrix[i, j]
                if weight > 0.01:
                    self.graph.add_edge(
                        well_ids[i], well_ids[j],
                        weight=float(weight),
                        distance=float(self.distance_matrix[i, j])
                    )

        return self.graph

    def spatial_interpolation(self, values: np.ndarray, grid_resolution: int = 50) -> Dict:
        lat_range = np.linspace(self.coords[:, 0].min() - 0.02, self.coords[:, 0].max() + 0.02, grid_resolution)
        lon_range = np.linspace(self.coords[:, 1].min() - 0.02, self.coords[:, 1].max() + 0.02, grid_resolution)
        grid_lat, grid_lon = np.meshgrid(lat_range, lon_range)

        grid_values = griddata(self.coords, values, (grid_lat, grid_lon), method="cubic")
        grid_values = np.nan_to_num(grid_values, nan=np.nanmean(values))

        return {
            "lat_grid": grid_lat.tolist(),
            "lon_grid": grid_lon.tolist(),
            "values": grid_values.tolist(),
            "lat_range": [float(lat_range.min()), float(lat_range.max())],
            "lon_range": [float(lon_range.min()), float(lon_range.max())]
        }

    def get_neighbors(self, well_idx: int, k: int = 5) -> List[int]:
        if self.distance_matrix is None:
            self.compute_distance_matrix()
        distances = self.distance_matrix[well_idx].copy()
        distances[well_idx] = np.inf
        return list(np.argsort(distances)[:k])

    def compute_spatial_features(self, production_values: np.ndarray) -> Dict[str, np.ndarray]:
        if self.adjacency_matrix is None:
            self.build_adjacency_graph()

        neighbor_avg = np.zeros(len(self.coords))
        spatial_lag = np.zeros(len(self.coords))
        local_moran = np.zeros(len(self.coords))

        global_mean = np.mean(production_values)

        for i in range(len(self.coords)):
            weights = self.adjacency_matrix[i]
            total_weight = weights.sum()
            if total_weight > 0:
                neighbor_avg[i] = np.sum(weights * production_values) / total_weight
                spatial_lag[i] = neighbor_avg[i] - production_values[i]
                zi = production_values[i] - global_mean
                wz = np.sum(weights * (production_values - global_mean)) / total_weight
                local_moran[i] = zi * wz

        return {
            "neighbor_avg_production": neighbor_avg,
            "spatial_lag": spatial_lag,
            "local_moran_i": local_moran
        }

    def get_edge_list(self) -> List[Dict]:
        if self.adjacency_matrix is None:
            self.build_adjacency_graph()
        edges = []
        well_ids = self.wells["well_id"].tolist()
        for i in range(len(well_ids)):
            for j in range(i + 1, len(well_ids)):
                if self.adjacency_matrix[i, j] > 0.01:
                    edges.append({
                        "source": well_ids[i],
                        "target": well_ids[j],
                        "weight": round(float(self.adjacency_matrix[i, j]), 4),
                        "distance": round(float(self.distance_matrix[i, j]), 6)
                    })
        return edges

    def get_graph_data_for_pyg(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.adjacency_matrix is None:
            self.build_adjacency_graph()
        edges = np.array(np.where(self.adjacency_matrix > 0.01))
        weights = self.adjacency_matrix[edges[0], edges[1]]
        return edges, weights
