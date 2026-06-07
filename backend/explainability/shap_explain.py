"""
Explainability Module: SHAP analysis + spatial influence explanations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class SpatialExplainer:
    def __init__(self, wells_df: pd.DataFrame, connectivity_matrix: np.ndarray):
        self.wells = wells_df
        self.connectivity = connectivity_matrix
        self.shap_values = None

    def compute_shap_values(self, model, X: np.ndarray, feature_names: List[str]) -> Dict:
        if not HAS_SHAP:
            return self._compute_permutation_importance(model, X, feature_names)

        explainer = shap.TreeExplainer(model)
        self.shap_values = explainer.shap_values(X[:500])

        importance = np.abs(self.shap_values).mean(axis=0)
        feature_importance = dict(zip(feature_names, importance.tolist()))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: -x[1]))

        return {
            "feature_importance": feature_importance,
            "top_features": list(feature_importance.keys())[:10],
            "num_samples_explained": min(500, len(X))
        }

    def explain_well_prediction(
        self, well_idx: int, model, X: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        neighbors = np.where(self.connectivity[well_idx] > 0.01)[0]
        neighbor_weights = self.connectivity[well_idx][neighbors]

        well_info = self.wells.iloc[well_idx]

        spatial_influence = []
        for i, (n_idx, weight) in enumerate(zip(neighbors, neighbor_weights)):
            spatial_influence.append({
                "neighbor_id": self.wells.iloc[n_idx]["well_id"],
                "distance": float(np.sqrt(
                    (well_info["latitude"] - self.wells.iloc[n_idx]["latitude"])**2
                    + (well_info["longitude"] - self.wells.iloc[n_idx]["longitude"])**2
                )),
                "connectivity_weight": float(weight),
                "same_zone": bool(well_info["reservoir_zone"] == self.wells.iloc[n_idx]["reservoir_zone"])
            })

        spatial_influence.sort(key=lambda x: -x["connectivity_weight"])

        if HAS_SHAP and self.shap_values is not None and well_idx < len(self.shap_values):
            well_shap = dict(zip(feature_names, self.shap_values[well_idx].tolist()))
        else:
            well_shap = {}

        return {
            "well_id": well_info["well_id"],
            "num_neighbors": len(neighbors),
            "spatial_influence": spatial_influence[:10],
            "feature_contributions": well_shap,
            "total_spatial_influence": float(neighbor_weights.sum()),
            "zone": well_info["reservoir_zone"]
        }

    def spatial_dependency_ranking(self) -> List[Dict]:
        rankings = []
        for i in range(len(self.wells)):
            total_connectivity = float(self.connectivity[i].sum())
            num_neighbors = int((self.connectivity[i] > 0.01).sum())
            rankings.append({
                "well_id": self.wells.iloc[i]["well_id"],
                "total_connectivity": round(total_connectivity, 4),
                "num_neighbors": num_neighbors,
                "avg_connectivity": round(total_connectivity / max(num_neighbors, 1), 4),
                "zone": self.wells.iloc[i]["reservoir_zone"]
            })
        rankings.sort(key=lambda x: -x["total_connectivity"])
        return rankings

    def _compute_permutation_importance(self, model, X: np.ndarray, feature_names: List[str]) -> Dict:
        from sklearn.metrics import mean_squared_error
        baseline_pred = model.predict(X[:500])
        importances = {}
        for i, name in enumerate(feature_names):
            X_permuted = X[:500].copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            perm_pred = model.predict(X_permuted)
            importances[name] = float(mean_squared_error(baseline_pred, perm_pred))
        importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
        return {
            "feature_importance": importances,
            "top_features": list(importances.keys())[:10],
            "method": "permutation"
        }
