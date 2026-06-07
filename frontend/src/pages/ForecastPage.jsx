import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Plot from 'react-plotly.js'
import { listWells, multiStepForecast } from '../api/client'

function ForecastPage() {
  const [wells, setWells] = useState([])
  const [selectedWell, setSelectedWell] = useState('')
  const [forecast, setForecast] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    listWells().then(res => {
      setWells(res.data.wells || [])
      if (res.data.wells?.length > 0) setSelectedWell(res.data.wells[0].well_id)
    })
  }, [])

  const runForecast = () => {
    if (!selectedWell) return
    setLoading(true)
    multiStepForecast(selectedWell)
      .then(res => { setForecast(res.data); setLoading(false) })
      .catch(() => setLoading(false))
  }

  useEffect(() => { if (selectedWell) runForecast() }, [selectedWell])

  const renderForecastChart = (key, data) => {
    if (!data?.forecast) return null
    const days = data.forecast.map(f => f.day)
    const predicted = data.forecast.map(f => f.predicted_rate)
    const lower = data.forecast.map(f => f.lower_bound)
    const upper = data.forecast.map(f => f.upper_bound)

    return (
      <div className="bg-oil-card border border-oil-border rounded-xl p-4">
        <h3 className="text-sm font-semibold mb-2 text-gray-300">Forecast: {key}</h3>
        <Plot
          data={[
            { x: days, y: upper, type: 'scatter', mode: 'lines', name: 'Upper 95%', line: { width: 0 }, showlegend: false },
            { x: days, y: lower, type: 'scatter', mode: 'lines', name: 'CI', fill: 'tonexty', fillcolor: 'rgba(0,212,170,0.1)', line: { width: 0 } },
            { x: days, y: predicted, type: 'scatter', mode: 'lines', name: 'Predicted', line: { color: '#00d4aa', width: 2 } },
          ]}
          layout={{
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { color: '#94a3b8', size: 10 },
            margin: { t: 10, b: 40, l: 50, r: 20 },
            height: 220,
            xaxis: { title: 'Days Ahead', gridcolor: '#1e293b' },
            yaxis: { title: 'bbl/d', gridcolor: '#1e293b' },
            legend: { orientation: 'h', y: -0.25 },
          }}
          config={{ responsive: true, displayModeBar: false }}
          className="w-full"
        />
        <div className="flex gap-4 mt-2 text-xs text-gray-400">
          <span>Current: <span className="text-white font-bold">{Math.round(data.current_rate)} bbl/d</span></span>
          <span>Decline: <span className="text-oil-warning font-bold">{data.avg_decline_rate}%/day</span></span>
        </div>
      </div>
    )
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">Multi-Step Production Forecast</h1>
          <p className="text-sm text-gray-400">Uncertainty-aware predictions at t+1, t+7, t+30 horizons</p>
        </div>
        <select
          value={selectedWell}
          onChange={e => setSelectedWell(e.target.value)}
          className="bg-oil-card border border-oil-border rounded-lg px-3 py-2 text-sm text-white"
        >
          {wells.map(w => <option key={w.well_id} value={w.well_id}>{w.well_id}</option>)}
        </select>
      </div>

      {loading && <div className="text-oil-accent animate-pulse">Running forecast models...</div>}

      {forecast && forecast.forecasts && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {Object.entries(forecast.forecasts).map(([key, data]) => (
            <div key={key}>{renderForecastChart(key, data)}</div>
          ))}
        </div>
      )}
    </motion.div>
  )
}

export default ForecastPage
