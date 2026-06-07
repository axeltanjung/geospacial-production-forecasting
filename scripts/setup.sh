#!/bin/bash
set -e

echo "========================================"
echo "  GEOSPATIAL PRODUCTION FORECASTING"
echo "  Setup Script"
echo "========================================"

if [ ! -d "venv" ]; then
    echo "[1/4] Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment already exists"
fi

echo "[2/4] Activating environment and installing dependencies..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "[3/4] Generating synthetic dataset..."
python -m backend.simulation.data_generator

echo "[4/4] Training models..."
python -m backend.training.pipeline

echo ""
echo "========================================"
echo "  SETUP COMPLETE"
echo "  Run: ./scripts/run.sh"
echo "========================================"
