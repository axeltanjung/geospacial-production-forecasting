#!/bin/bash
echo "========================================"
echo "  STARTING GEOSPATIAL AI PLATFORM"
echo "========================================"

if [ ! -d "data/generated" ]; then
    echo "Data not found. Running setup first..."
    ./scripts/setup.sh
fi

source venv/bin/activate

echo "[Backend] Starting FastAPI on port 8000..."
python backend/api/main.py &
BACKEND_PID=$!

echo "[Frontend] Starting React dev server on port 5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "========================================"
echo "  SERVICES RUNNING"
echo "  Backend API: http://localhost:8000"
echo "  API Docs:    http://localhost:8000/docs"
echo "  Frontend:    http://localhost:5173"
echo "========================================"
echo "  Press Ctrl+C to stop all services"
echo ""

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
