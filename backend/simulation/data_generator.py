"""
Synthetic Geospatial Oil & Gas Production Data Generator

Generates realistic reservoir production data with:
- Spatially distributed wells across a field
- Reservoir zones with distinct properties
- Fault lines reducing connectivity
- Temporal production with decline, interference, and events
- 200,000+ rows of time-series data
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pathlib import Path
import json


class ReservoirFieldGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.num_wells = 150
        self.num_timesteps = 1460  # ~4 years of daily data
        self.num_zones = 5
        self.num_faults = 8
        self.field_center_lat = 31.5
        self.field_center_lon = -103.5
        self.field_extent = 0.3

    def generate_fault_lines(self):
        faults = []
        for i in range(self.num_faults):
            start_lat = self.field_center_lat + np.random.uniform(-self.field_extent, self.field_extent)
            start_lon = self.field_center_lon + np.random.uniform(-self.field_extent, self.field_extent)
            angle = np.random.uniform(0, np.pi)
            length = np.random.uniform(0.05, 0.15)
            end_lat = start_lat + length * np.sin(angle)
            end_lon = start_lon + length * np.cos(angle)
            transmissibility = np.random.uniform(0.0, 0.3)
            faults.append({
                "fault_id": f"F{i+1:03d}",
                "start_lat": start_lat,
                "start_lon": start_lon,
                "end_lat": end_lat,
                "end_lon": end_lon,
                "transmissibility": transmissibility
            })
        self.faults = faults
        return faults

    def generate_wells(self):
        wells = []
        zone_centers = [
            (self.field_center_lat + np.random.uniform(-0.15, 0.15),
             self.field_center_lon + np.random.uniform(-0.15, 0.15))
            for _ in range(self.num_zones)
        ]

        for i in range(self.num_wells):
            zone_idx = np.random.randint(0, self.num_zones)
            center = zone_centers[zone_idx]
            lat = center[0] + np.random.normal(0, 0.04)
            lon = center[1] + np.random.normal(0, 0.04)

            dist_to_fault = self._min_distance_to_fault(lat, lon)
            depth = np.random.uniform(5000, 12000)
            permeability = np.random.lognormal(mean=3.0, sigma=0.8)
            porosity = np.clip(np.random.normal(0.15, 0.04), 0.05, 0.35)

            wells.append({
                "well_id": f"W{i+1:04d}",
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "depth": round(depth, 1),
                "reservoir_zone": f"Zone_{zone_idx + 1}",
                "distance_to_fault_line": round(dist_to_fault, 4),
                "distance_to_boundary": round(np.random.uniform(0.01, 0.25), 4),
                "permeability_index": round(permeability, 2),
                "porosity_index": round(porosity, 4),
                "initial_pressure": round(np.random.uniform(3000, 6000), 1),
                "initial_production": round(np.random.uniform(200, 2000), 1),
                "completion_date_offset": np.random.randint(0, 365)
            })

        self.wells = pd.DataFrame(wells)
        self.well_coords = self.wells[["latitude", "longitude"]].values
        self.distance_matrix = cdist(self.well_coords, self.well_coords, metric="euclidean")
        return self.wells

    def _min_distance_to_fault(self, lat, lon):
        if not hasattr(self, "faults") or not self.faults:
            return 1.0
        min_dist = float("inf")
        for fault in self.faults:
            dist = self._point_to_line_distance(
                lat, lon,
                fault["start_lat"], fault["start_lon"],
                fault["end_lat"], fault["end_lon"]
            )
            min_dist = min(min_dist, dist)
        return min_dist

    @staticmethod
    def _point_to_line_distance(px, py, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def _compute_connectivity(self):
        connectivity = np.exp(-self.distance_matrix / 0.05)
        for i in range(self.num_wells):
            for j in range(i + 1, self.num_wells):
                for fault in self.faults:
                    if self._line_crosses_fault(i, j, fault):
                        connectivity[i, j] *= fault["transmissibility"]
                        connectivity[j, i] *= fault["transmissibility"]
        np.fill_diagonal(connectivity, 0)
        self.connectivity = connectivity
        return connectivity

    def _line_crosses_fault(self, i, j, fault):
        x1, y1 = self.well_coords[i]
        x2, y2 = self.well_coords[j]
        x3, y3 = fault["start_lat"], fault["start_lon"]
        x4, y4 = fault["end_lat"], fault["end_lon"]
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return False
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        return 0 <= t <= 1 and 0 <= u <= 1

    def generate_temporal_data(self):
        self._compute_connectivity()
        all_records = []
        base_date = pd.Timestamp("2020-01-01")

        pressures = np.array([w["initial_pressure"] for _, w in self.wells.iterrows()])
        productions = np.array([w["initial_production"] for _, w in self.wells.iterrows()])

        for t in range(self.num_timesteps):
            timestamp = base_date + pd.Timedelta(days=t)
            seasonal_factor = 1.0 + 0.03 * np.sin(2 * np.pi * t / 365)

            neighbor_pressure_effect = np.zeros(self.num_wells)
            for i in range(self.num_wells):
                neighbors = self.connectivity[i]
                weighted_pressure = np.sum(neighbors * pressures) / (np.sum(neighbors) + 1e-8)
                neighbor_pressure_effect[i] = 0.01 * (weighted_pressure - pressures[i])

            decline_rates = np.array([
                0.0005 + 0.0002 * (1 - self.wells.iloc[i]["porosity_index"] / 0.35)
                for i in range(self.num_wells)
            ])

            pressures += neighbor_pressure_effect
            pressures -= decline_rates * pressures * 0.01
            pressures = np.clip(pressures, 500, 7000)

            production_noise = np.random.normal(0, 0.02, self.num_wells)
            productions = (
                productions
                * (1 - decline_rates)
                * seasonal_factor
                * (1 + production_noise)
                * (pressures / (pressures + 1000))
            )

            for i in range(self.num_wells):
                if np.random.random() < 0.001:
                    productions[i] *= np.random.uniform(0.3, 0.6)
                if np.random.random() < 0.0005:
                    productions[i] *= np.random.uniform(1.3, 1.8)
                    pressures[i] *= np.random.uniform(1.05, 1.15)

            productions = np.clip(productions, 10, 5000)

            neighbor_avg_prod = np.array([
                np.sum(self.connectivity[i] * productions) / (np.sum(self.connectivity[i]) + 1e-8)
                for i in range(self.num_wells)
            ])
            neighbor_avg_pressure = np.array([
                np.sum(self.connectivity[i] * pressures) / (np.sum(self.connectivity[i]) + 1e-8)
                for i in range(self.num_wells)
            ])

            for i in range(self.num_wells):
                well = self.wells.iloc[i]
                if t < well["completion_date_offset"]:
                    continue

                water_cut = np.clip(0.1 + 0.0003 * t + np.random.normal(0, 0.02), 0, 0.95)
                gor = np.clip(500 + 0.5 * t + np.random.normal(0, 30), 100, 5000)
                injection_rate = max(0, np.random.normal(100, 30)) if np.random.random() < 0.3 else 0

                spatial_density = np.sum(self.distance_matrix[i] < 0.05) / self.num_wells
                local_depletion = 1.0 - (pressures[i] / well["initial_pressure"])

                record = {
                    "well_id": well["well_id"],
                    "timestamp": timestamp,
                    "production_rate": round(productions[i], 2),
                    "water_cut": round(water_cut, 4),
                    "gas_oil_ratio": round(gor, 1),
                    "reservoir_pressure": round(pressures[i], 1),
                    "injection_rate": round(injection_rate, 1),
                    "neighbor_well_avg_production": round(neighbor_avg_prod[i], 2),
                    "neighbor_well_pressure_avg": round(neighbor_avg_pressure[i], 1),
                    "spatial_density_index": round(spatial_density, 4),
                    "local_depletion_factor": round(local_depletion, 4),
                }
                all_records.append(record)

            if t % 100 == 0:
                print(f"  Generated timestep {t}/{self.num_timesteps}")

        self.temporal_data = pd.DataFrame(all_records)
        return self.temporal_data

    def engineer_features(self, df):
        df = df.sort_values(["well_id", "timestamp"]).reset_index(drop=True)
        df["decline_rate"] = df.groupby("well_id")["production_rate"].pct_change()
        df["production_momentum"] = df.groupby("well_id")["production_rate"].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        df["pressure_diffusion_index"] = (
            df["neighbor_well_pressure_avg"] - df["reservoir_pressure"]
        ) / (df["reservoir_pressure"] + 1e-8)

        df["production_volatility"] = df.groupby("well_id")["production_rate"].transform(
            lambda x: x.rolling(window=30, min_periods=1).std()
        )

        df["decline_rate"] = df["decline_rate"].fillna(0).clip(-0.5, 0.5)
        df["production_volatility"] = df["production_volatility"].fillna(0)

        mask = np.random.random(len(df)) < 0.02
        df.loc[mask, "reservoir_pressure"] = np.nan
        mask2 = np.random.random(len(df)) < 0.01
        df.loc[mask2, "water_cut"] = np.nan

        return df

    def generate_targets(self, df):
        df = df.sort_values(["well_id", "timestamp"]).reset_index(drop=True)
        df["future_production_t1"] = df.groupby("well_id")["production_rate"].shift(-1)
        df["future_production_t7"] = df.groupby("well_id")["production_rate"].shift(-7)
        df["future_production_t30"] = df.groupby("well_id")["production_rate"].shift(-30)

        df["reservoir_health_score"] = (
            0.4 * (df["reservoir_pressure"] / 6000).clip(0, 1)
            + 0.3 * (1 - df["water_cut"].fillna(0))
            + 0.3 * (df["production_rate"] / df.groupby("well_id")["production_rate"].transform("max"))
        ).round(4)

        return df

    def run(self, output_dir="data/generated"):
        print("=" * 60)
        print("GEOSPATIAL OIL & GAS DATA GENERATOR")
        print("=" * 60)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n[1/6] Generating fault lines...")
        faults = self.generate_fault_lines()

        print("[2/6] Generating well locations...")
        wells = self.generate_wells()
        print(f"  Generated {len(wells)} wells across {self.num_zones} zones")

        print("[3/6] Generating temporal production data...")
        temporal = self.generate_temporal_data()
        print(f"  Generated {len(temporal):,} temporal records")

        print("[4/6] Engineering features...")
        temporal = self.engineer_features(temporal)

        print("[5/6] Generating forecast targets...")
        temporal = self.generate_targets(temporal)

        print("[6/6] Saving datasets...")

        wells.to_csv(output_path / "wells_static.csv", index=False)
        temporal.to_csv(output_path / "production_temporal.csv", index=False)

        with open(output_path / "fault_lines.json", "w") as f:
            json.dump(faults, f, indent=2)

        connectivity_df = pd.DataFrame(
            self.connectivity,
            index=wells["well_id"],
            columns=wells["well_id"]
        )
        connectivity_df.to_csv(output_path / "connectivity_matrix.csv")

        metadata = {
            "num_wells": self.num_wells,
            "num_timesteps": self.num_timesteps,
            "num_zones": self.num_zones,
            "num_faults": self.num_faults,
            "total_records": len(temporal),
            "date_range": [str(temporal["timestamp"].min()), str(temporal["timestamp"].max())],
            "field_center": [self.field_center_lat, self.field_center_lon]
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'=' * 60}")
        print("GENERATION COMPLETE")
        print(f"  Total records: {len(temporal):,}")
        print(f"  Wells: {self.num_wells}")
        print(f"  Time steps: {self.num_timesteps}")
        print(f"  Output: {output_path.resolve()}")
        print(f"{'=' * 60}")

        return wells, temporal, faults


if __name__ == "__main__":
    generator = ReservoirFieldGenerator(seed=42)
    generator.run()
