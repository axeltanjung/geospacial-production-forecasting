"""
Gaussian Process Regression for Spatial Production Interpolation.

Uses GPyTorch for efficient GP inference with:
- RBF + Matérn composite kernel for multi-scale spatial patterns
- Uncertainty quantification via posterior variance
- Spatial production surface mapping
"""

import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

try:
    import gpytorch
    from gpytorch.models import ExactGP
    from gpytorch.means import ConstantMean
    from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, AdditiveKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.distributions import MultivariateNormal
    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False


class SpatialGPModel(gpytorch.models.ExactGP if HAS_GPYTORCH else object):
    def __init__(self, train_x, train_y, likelihood):
        if not HAS_GPYTORCH:
            raise ImportError("GPyTorch required. Install via: pip install gpytorch")
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        rbf_kernel = ScaleKernel(RBFKernel(ard_num_dims=2))
        matern_kernel = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=2))
        self.covar_module = AdditiveKernel(rbf_kernel, matern_kernel)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


class SpatialGaussianProcess:
    def __init__(self, learning_rate: float = 0.1, training_iterations: int = 100):
        self.lr = learning_rate
        self.iterations = training_iterations
        self.model = None
        self.likelihood = None
        self.train_x = None
        self.train_y = None
        self.is_trained = False

    def fit(self, coordinates: np.ndarray, values: np.ndarray) -> Dict:
        if not HAS_GPYTORCH:
            return self._fit_fallback(coordinates, values)

        self.train_x = torch.tensor(coordinates, dtype=torch.float32)
        self.train_y = torch.tensor(values, dtype=torch.float32)

        self.likelihood = GaussianLikelihood()
        self.model = SpatialGPModel(self.train_x, self.train_y, self.likelihood)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        losses = []
        for i in range(self.iterations):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.is_trained = True
        return {
            "final_loss": losses[-1],
            "iterations": self.iterations,
            "num_training_points": len(values)
        }

    def predict(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        if not HAS_GPYTORCH:
            return self._predict_fallback(coordinates)

        test_x = torch.tensor(coordinates, dtype=torch.float32)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))
            mean = observed_pred.mean.numpy()
            lower, upper = observed_pred.confidence_region()

        return mean, lower.numpy(), upper.numpy()

    def predict_grid(self, lat_range: Tuple[float, float], lon_range: Tuple[float, float],
                     resolution: int = 50) -> Dict:
        lat_grid = np.linspace(lat_range[0], lat_range[1], resolution)
        lon_grid = np.linspace(lon_range[0], lon_range[1], resolution)
        grid_lat, grid_lon = np.meshgrid(lat_grid, lon_grid)
        grid_points = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])

        mean, lower, upper = self.predict(grid_points)

        return {
            "lat_grid": grid_lat.tolist(),
            "lon_grid": grid_lon.tolist(),
            "mean": mean.reshape(resolution, resolution).tolist(),
            "lower": lower.reshape(resolution, resolution).tolist(),
            "upper": upper.reshape(resolution, resolution).tolist(),
            "uncertainty": (upper - lower).reshape(resolution, resolution).tolist()
        }

    def _fit_fallback(self, coordinates: np.ndarray, values: np.ndarray) -> Dict:
        from scipy.interpolate import RBFInterpolator
        self._rbf = RBFInterpolator(coordinates, values, kernel="thin_plate_spline")
        self._train_std = np.std(values)
        self._train_coords = coordinates
        self._train_values = values
        self.is_trained = True
        return {"final_loss": 0.0, "iterations": 0, "num_training_points": len(values), "fallback": True}

    def _predict_fallback(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from scipy.spatial.distance import cdist
        mean = self._rbf(coordinates)
        dists = cdist(coordinates, self._train_coords)
        min_dists = dists.min(axis=1)
        uncertainty = self._train_std * (1 - np.exp(-min_dists / 0.05))
        lower = mean - 1.96 * uncertainty
        upper = mean + 1.96 * uncertainty
        return mean, lower, upper
