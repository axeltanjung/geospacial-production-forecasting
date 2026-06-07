# Geospatial Oil & Gas Production Forecasting Platform

## Spatio-Temporal AI System for Reservoir Analytics

A production-grade AI platform that forecasts oil/gas production per well by explicitly modeling spatial dependency between wells using Graph Neural Networks, Gaussian Processes, and Spatio-Temporal deep learning.

---
## Project Preview

<img src="assets/Geo Mapping.png" width="1200"/>

## Project Background

### Why Spatial Dependency Matters in Reservoirs

Oil and gas reservoirs are continuous geological formations where fluid flow connects neighboring wells through shared rock properties. **Production is NOT independent** — nearby wells influence each other due to:

- **Pressure communication**: Drawing fluid from one well reduces pressure at neighbors
- **Reservoir connectivity**: Permeable rock allows fluid migration between well drainage areas
- **Interference effects**: High-rate producers can starve offset wells of supply
- **Shared aquifer support**: Water influx from common aquifers affects multiple wells

Traditional production forecasting treats each well independently (decline curve analysis), ignoring these critical spatial interactions.

### Production Interference Effect

When Well A increases production rate:
1. Local reservoir pressure drops around Well A
2. Pressure gradient steepens toward Well A from neighboring wells
3. Wells B, C nearby experience accelerated pressure decline
4. Production at B, C decreases even without operational changes

This platform **explicitly models these spatial dependencies** using graph-based neural networks.

### Reservoir Connectivity Physics

- **Darcy's Law**: Flow rate proportional to pressure gradient and permeability
- **Pressure Diffusion**: Pressure disturbances propagate through reservoir at rate dependent on hydraulic diffusivity
- **Fault Barriers**: Geological faults reduce or eliminate inter-well connectivity
- **Permeability Heterogeneity**: Spatial variation in rock quality creates preferential flow paths

---

## Business Value

| Metric | Impact |
|--------|--------|
| Drilling Planning | Optimize well placement considering neighbor interference |
| Production Optimization | Allocate rates across wells to maximize field output |
| Reservoir Management | Early detection of connectivity changes and depletion |
| Reduced Production Loss | Predict and prevent interference-driven decline |
| Capital Efficiency | Avoid drilling in depleted zones identified by spatial model |

---

## AI/ML Architecture

### Model 1: Gaussian Process Regression
- **Purpose**: Spatial interpolation + uncertainty quantification
- **Kernel**: Composite RBF + Matérn for capturing multi-scale spatial patterns
- **Output**: Production surface map with confidence intervals
- **Strength**: Principled uncertainty bounds for risk assessment

### Model 2: Graph Neural Network (GCN + GAT)
- **Purpose**: Learn well-to-well production dependencies
- **Architecture**: Graph Convolutional + Graph Attention layers
- **Graph**: Wells as nodes, spatial proximity as edges, weighted by connectivity
- **Output**: Node-level production forecast informed by neighbor states

### Model 3: Spatio-Temporal GNN (ST-GNN)
- **Purpose**: Jointly model temporal evolution and spatial dependency
- **Architecture**: LSTM temporal encoder → GNN spatial aggregation → prediction head
- **Captures**: How production changes propagate through the field over time

### Baseline: XGBoost + LightGBM
- **Purpose**: Strong ML baseline for comparison
- **Features**: Engineered spatial + temporal + static well features

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (React + Vite)                    │
│  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌──────────────┐  │
│  │ Geo Map  │  │ Forecast │  │Network │  │  AI Insights │  │
│  │Dashboard │  │  Charts  │  │ Graph  │  │  SHAP/XAI    │  │
│  └────┬─────┘  └────┬─────┘  └───┬────┘  └──────┬───────┘  │
└───────┼──────────────┼────────────┼───────────────┼──────────┘
        │              │            │               │
        ▼              ▼            ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Predict  │  │  Map     │  │  Graph   │  │  Explain   │  │
│  │ Service  │  │  Data    │  │Structure │  │  Service   │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘  │
└───────┼──────────────┼─────────────┼──────────────┼──────────┘
        │              │             │              │
        ▼              ▼             ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML Engine Layer                            │
│  ┌─────┐  ┌──────────────┐  ┌───────┐  ┌────────────────┐  │
│  │ GNN │  │Gaussian Proc.│  │ST-GNN │  │ XGBoost/LGBM  │  │
│  └──┬──┘  └──────┬───────┘  └───┬───┘  └───────┬────────┘  │
└─────┼─────────────┼─────────────┼───────────────┼───────────┘
      │             │             │               │
      ▼             ▼             ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│              Geospatial Engine + Graph Builder                │
│  Distance Matrix │ Adjacency Graph │ Spatial Interpolation   │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Pandas, NumPy |
| ML Models | PyTorch, PyTorch Geometric, GPyTorch, XGBoost, LightGBM |
| Geospatial | GeoPandas, Shapely, SciPy spatial, NetworkX |
| Explainability | SHAP |
| Frontend | React 18, Vite, TailwindCSS |
| Mapping | Leaflet, react-leaflet, leaflet-heat |
| Charts | Plotly.js, D3.js |
| Animation | Framer Motion |
| CI/CD | GitHub Actions |

---

## Quick Start (GitHub Codespaces / Local)

### Prerequisites
- Python 3.10+
- Node.js 18+
- Git

### Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/geospatial-production-forecasting.git
cd geospatial-production-forecasting

# Backend setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Generate synthetic data
python -m backend.simulation.data_generator

# Train models
python -m backend.training.pipeline

# Start backend
python backend/api/main.py

# Frontend setup (new terminal)
cd frontend
npm install
npm run dev
```

### Access
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend: http://localhost:5173

### One-Command Start
```bash
chmod +x scripts/setup.sh scripts/run.sh
./scripts/setup.sh  # First time only
./scripts/run.sh    # Start both servers
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/predict/production` | POST | Single well production forecast |
| `/predict/spatial-impact` | POST | Spatial dependency prediction |
| `/well/{id}` | GET | Well details and history |
| `/map/data` | GET | Geospatial map data (all wells) |
| `/graph/structure` | GET | Well connectivity graph |
| `/forecast/multi-step` | POST | Multi-horizon forecast (t+1, t+7, t+30) |
| `/explain/spatial` | POST | SHAP + spatial influence explanation |

---

## Domain Knowledge

### Decline Curve Analysis
Production naturally declines over time following:
- **Exponential**: q(t) = qi × e^(-Dt)
- **Hyperbolic**: q(t) = qi / (1 + bDit)^(1/b)
- **Harmonic**: q(t) = qi / (1 + Dit)

### Spatial Autocorrelation
Wells close in space tend to have correlated production due to shared reservoir properties (Tobler's First Law of Geography applied to subsurface).

### Fault Line Impact
Geological faults act as barriers or conduits:
- **Sealing faults**: Block pressure communication
- **Leaking faults**: Partially transmit pressure
- **Conduit faults**: Enhance connectivity along fault plane

---

## Future Improvements

- Real satellite/seismic reservoir mapping integration
- Reinforcement learning for production rate optimization
- Digital twin reservoir simulation coupling
- Real-time sensor data ingestion
- Transfer learning across different oil fields
- 3D subsurface visualization
- Automated well placement optimization

---

## License

MIT License

---

## Author

Built as a portfolio project demonstrating expertise in:
- Machine Learning Engineering
- Geospatial Data Science
- Reservoir Engineering Analytics
- MLOps Architecture
- Full-Stack AI System Development
