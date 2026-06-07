"""
Training Pipeline: Orchestrates data loading, model training, and evaluation.
"""

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from typing import Dict, Optional

from backend.utils.config import DATA_DIR, MODELS_DIR, get_logger
from backend.geospatial.engine import GeospatialEngine
from backend.models.baseline_ml.model import BaselineModels, FEATURE_COLUMNS
from backend.models.gaussian_process.model import SpatialGaussianProcess
from backend.models.gnn.model import WellProductionGNN
from backend.models.spatio_temporal.model import SpatioTemporalGNN, STGNNTrainer

logger = get_logger("training")


class TrainingPipeline:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.models_dir = MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.wells = None
        self.temporal = None
        self.geo_engine = None
        self.results = {}

    def load_data(self):
        logger.info("Loading datasets...")
        self.wells = pd.read_csv(self.data_dir / "wells_static.csv")
        self.temporal = pd.read_csv(self.data_dir / "production_temporal.csv")
        self.temporal["timestamp"] = pd.to_datetime(self.temporal["timestamp"])

        with open(self.data_dir / "fault_lines.json") as f:
            self.faults = json.load(f)

        logger.info(f"  Loaded {len(self.wells)} wells, {len(self.temporal):,} temporal records")

    def setup_geospatial(self):
        logger.info("Setting up geospatial engine...")
        self.geo_engine = GeospatialEngine(self.wells)
        self.geo_engine.compute_distance_matrix()
        self.geo_engine.build_adjacency_graph(self.faults)
        logger.info(f"  Adjacency graph built: {(self.geo_engine.adjacency_matrix > 0.01).sum()} edges")

    def train_baseline(self) -> Dict:
        logger.info("Training baseline models (XGBoost + LightGBM)...")
        baseline = BaselineModels()
        X, y = baseline.prepare_data(self.temporal, self.wells, target_col="future_production_t1")
        results = baseline.train(X, y)
        self.results["baseline"] = results

        import joblib
        if baseline.xgb_model:
            joblib.dump(baseline.xgb_model, self.models_dir / "xgboost_model.joblib")
        if baseline.lgb_model:
            joblib.dump(baseline.lgb_model, self.models_dir / "lightgbm_model.joblib")
        joblib.dump(baseline.scaler, self.models_dir / "scaler.joblib")

        logger.info(f"  Baseline results: {results}")
        return results

    def train_gaussian_process(self) -> Dict:
        logger.info("Training Gaussian Process (spatial interpolation)...")
        coords = self.wells[["latitude", "longitude"]].values

        latest = self.temporal.groupby("well_id").last().reset_index()
        merged = latest.merge(self.wells[["well_id", "latitude", "longitude"]], on="well_id")
        values = merged["production_rate"].values

        gp = SpatialGaussianProcess(learning_rate=0.1, training_iterations=50)
        results = gp.fit(coords, values)
        self.results["gaussian_process"] = results

        mean_pred, lower, upper = gp.predict(coords)
        mae = float(np.mean(np.abs(values - mean_pred)))
        coverage = float(np.mean((values >= lower) & (values <= upper)))
        results["prediction_mae"] = round(mae, 4)
        results["interval_coverage"] = round(coverage, 4)

        torch.save({"coords": coords, "values": values}, self.models_dir / "gp_training_data.pt")
        logger.info(f"  GP results: MAE={mae:.2f}, Coverage={coverage:.2%}")
        return results

    def train_gnn(self, epochs: int = 50) -> Dict:
        logger.info("Training GNN model...")

        latest = self.temporal.groupby("well_id").last().reset_index()
        merged = latest.merge(self.wells, on="well_id")

        feature_cols = [c for c in FEATURE_COLUMNS if c in merged.columns]
        X = torch.tensor(merged[feature_cols].fillna(0).values, dtype=torch.float32)
        adj = torch.tensor(self.geo_engine.adjacency_matrix, dtype=torch.float32)

        targets = merged[["future_production_t1", "future_production_t7", "future_production_t30"]].fillna(0).values
        y = torch.tensor(targets, dtype=torch.float32)

        model = WellProductionGNN(
            input_dim=X.shape[1],
            hidden_dim=128,
            output_dim=3,
            dropout=0.2
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        losses = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            mean, std = model(X, adj)
            loss = torch.nn.functional.mse_loss(mean, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            pred_mean, pred_std = model(X, adj)
        mae = float(torch.nn.functional.l1_loss(pred_mean, y))
        rmse = float(torch.sqrt(torch.nn.functional.mse_loss(pred_mean, y)))

        torch.save(model.state_dict(), self.models_dir / "gnn_model.pt")

        results = {"final_loss": losses[-1], "mae": round(mae, 4), "rmse": round(rmse, 4), "epochs": epochs}
        self.results["gnn"] = results
        logger.info(f"  GNN results: MAE={mae:.4f}, RMSE={rmse:.4f}")
        return results

    def train_stgnn(self, epochs: int = 30) -> Dict:
        logger.info("Training Spatio-Temporal GNN...")

        seq_length = 30
        well_ids = self.wells["well_id"].tolist()
        temporal_features_list = []
        spatial_features_list = []
        targets_list = []

        for wid in well_ids[:50]:
            well_data = self.temporal[self.temporal["well_id"] == wid].sort_values("timestamp")
            if len(well_data) < seq_length + 30:
                continue

            temporal_cols = [
                "production_rate", "water_cut", "gas_oil_ratio",
                "reservoir_pressure", "injection_rate"
            ]
            available = [c for c in temporal_cols if c in well_data.columns]
            seq = well_data[available].fillna(0).values[-seq_length - 30:-30]
            temporal_features_list.append(seq)

            spatial_cols = [
                "neighbor_well_avg_production", "neighbor_well_pressure_avg",
                "spatial_density_index", "local_depletion_factor"
            ]
            available_s = [c for c in spatial_cols if c in well_data.columns]
            spatial_features_list.append(well_data[available_s].fillna(0).values[-31])
            targets_list.append(well_data["production_rate"].values[-1])

        if not temporal_features_list:
            logger.warning("  Insufficient data for ST-GNN training")
            return {"status": "skipped"}

        temporal_input = torch.tensor(np.array(temporal_features_list), dtype=torch.float32)
        spatial_input = torch.tensor(np.array(spatial_features_list), dtype=torch.float32)
        target = torch.tensor(np.array(targets_list), dtype=torch.float32).unsqueeze(-1)

        n = len(temporal_features_list)
        adj = torch.tensor(self.geo_engine.adjacency_matrix[:n, :n], dtype=torch.float32)

        model = SpatioTemporalGNN(
            temporal_input_dim=temporal_input.shape[-1],
            spatial_input_dim=spatial_input.shape[-1],
            output_dim=1
        )
        trainer = STGNNTrainer(model)

        for epoch in range(epochs):
            loss = trainer.train_step(temporal_input, spatial_input, adj, target)
            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

        metrics = trainer.evaluate(temporal_input, spatial_input, adj, target)
        torch.save(model.state_dict(), self.models_dir / "stgnn_model.pt")

        self.results["stgnn"] = metrics
        logger.info(f"  ST-GNN results: {metrics}")
        return metrics

    def run(self):
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE START")
        logger.info("=" * 60)

        self.load_data()
        self.setup_geospatial()
        self.train_baseline()
        self.train_gaussian_process()
        self.train_gnn()
        self.train_stgnn()

        with open(self.models_dir / "training_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"  Models saved to: {self.models_dir}")
        logger.info("=" * 60)
        return self.results


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
