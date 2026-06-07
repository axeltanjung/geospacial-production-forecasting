"""
Baseline ML Models: XGBoost and LightGBM for production forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


FEATURE_COLUMNS = [
    "depth", "permeability_index", "porosity_index",
    "distance_to_fault_line", "distance_to_boundary",
    "water_cut", "gas_oil_ratio", "reservoir_pressure",
    "injection_rate", "neighbor_well_avg_production",
    "neighbor_well_pressure_avg", "spatial_density_index",
    "local_depletion_factor", "decline_rate",
    "production_momentum", "pressure_diffusion_index"
]


class BaselineModels:
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.metrics = {}

    def prepare_data(self, temporal_df: pd.DataFrame, wells_df: pd.DataFrame,
                     target_col: str = "future_production_t1") -> Tuple[np.ndarray, np.ndarray]:
        merged = temporal_df.merge(
            wells_df[["well_id", "depth", "permeability_index",
                      "porosity_index", "distance_to_fault_line",
                      "distance_to_boundary"]],
            on="well_id", how="left")
        available_cols = [c for c in FEATURE_COLUMNS if c in merged.columns]
        data = merged.dropna(subset=[target_col])
        X = data[available_cols].fillna(0).values
        y = data[target_col].values
        return X, y

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        results = {}

        if HAS_XGB:
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1
            )
            self.xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
            xgb_pred = self.xgb_model.predict(X_test_scaled)
            results["xgboost"] = self._compute_metrics(y_test, xgb_pred)

        if HAS_LGB:
            self.lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            self.lgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)])
            lgb_pred = self.lgb_model.predict(X_test_scaled)
            results["lightgbm"] = self._compute_metrics(y_test, lgb_pred)

        self.is_fitted = True
        self.metrics = results
        return results

    def predict(self, X: np.ndarray, model: str = "xgboost") -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Models must be trained first")
        X_scaled = self.scaler.transform(X)
        if model == "xgboost" and self.xgb_model:
            return self.xgb_model.predict(X_scaled)
        elif model == "lightgbm" and self.lgb_model:
            return self.lgb_model.predict(X_scaled)
        raise ValueError(f"Model {model} not available")

    def get_feature_importance(self, model: str = "xgboost") -> Dict[str, float]:
        if model == "xgboost" and self.xgb_model:
            importance = self.xgb_model.feature_importances_
        elif model == "lightgbm" and self.lgb_model:
            importance = self.lgb_model.feature_importances_
        else:
            return {}
        available_cols = FEATURE_COLUMNS[:len(importance)]
        return dict(sorted(zip(available_cols, importance.tolist()), key=lambda x: -x[1]))

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        return {
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4),
            "mape": round(float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100), 2)
        }
