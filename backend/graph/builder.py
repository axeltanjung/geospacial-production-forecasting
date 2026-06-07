from __future__ import annotations

"""
Graph Builder: Constructs well connectivity graphs for GNN models.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Tuple


class WellGraphBuilder:
    def __init__(self, wells_df: pd.DataFrame, distance_matrix: np.ndarray):
        self.wells = wells_df
        self.distance_matrix = distance_matrix
        self.num_wells = len(wells_df)

    def build_knn_graph(self, k: int = 5) -> nx.Graph:
        G = nx.Graph()
        well_ids = self.wells["well_id"].tolist()

        for i, wid in enumerate(well_ids):
            G.add_node(wid, idx=i)

        for i in range(self.num_wells):
            distances = self.distance_matrix[i].copy()
            distances[i] = np.inf
            nearest = np.argsort(distances)[:k]
            for j in nearest:
                if not G.has_edge(well_ids[i], well_ids[j]):
                    G.add_edge(well_ids[i], well_ids[j], weight=1.0 / (distances[j] + 1e-6))

        return G

    def build_threshold_graph(self, threshold: float = 0.05) -> nx.Graph:
        G = nx.Graph()
        well_ids = self.wells["well_id"].tolist()

        for i, wid in enumerate(well_ids):
            G.add_node(wid, idx=i)

        for i in range(self.num_wells):
            for j in range(i + 1, self.num_wells):
                if self.distance_matrix[i, j] < threshold:
                    weight = np.exp(-self.distance_matrix[i, j] / (threshold / 3))
                    G.add_edge(well_ids[i], well_ids[j], weight=weight)

        return G

    def build_zone_aware_graph(self, threshold: float = 0.05, zone_bonus: float = 1.5) -> nx.Graph:
        G = self.build_threshold_graph(threshold)
        well_ids = self.wells["well_id"].tolist()

        for i in range(self.num_wells):
            for j in range(i + 1, self.num_wells):
                if G.has_edge(well_ids[i], well_ids[j]):
                    if self.wells.iloc[i]["reservoir_zone"] == self.wells.iloc[j]["reservoir_zone"]:
                        G[well_ids[i]][well_ids[j]]["weight"] *= zone_bonus

        return G

    def get_edge_index(self, graph: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
        well_id_to_idx = {wid: i for i, wid in enumerate(self.wells["well_id"])}
        sources = []
        targets = []
        weights = []

        for u, v, data in graph.edges(data=True):
            src_idx = well_id_to_idx[u]
            tgt_idx = well_id_to_idx[v]
            sources.extend([src_idx, tgt_idx])
            targets.extend([tgt_idx, src_idx])
            w = data.get("weight", 1.0)
            weights.extend([w, w])

        edge_index = np.array([sources, targets], dtype=np.int64)
        edge_weights = np.array(weights, dtype=np.float32)
        return edge_index, edge_weights

    def get_graph_stats(self, graph: nx.Graph) -> Dict:
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "avg_degree": np.mean([d for _, d in graph.degree()]),
            "density": nx.density(graph),
            "num_components": nx.number_connected_components(graph),
            "avg_clustering": nx.average_clustering(graph, weight="weight")
        }

    def export_for_visualization(self, graph: nx.Graph) -> Dict:
        nodes = []
        for _, row in self.wells.iterrows():
            nodes.append({
                "id": row["well_id"],
                "lat": row["latitude"],
                "lon": row["longitude"],
                "zone": row["reservoir_zone"],
                "depth": row["depth"]
            })

        edges = []
        for u, v, data in graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "weight": round(data.get("weight", 1.0), 4)
            })

        return {"nodes": nodes, "edges": edges}
