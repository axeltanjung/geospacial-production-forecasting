import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import Plot from 'react-plotly.js'
import { getWellDetail, predictSpatialImpact } from '../api/client'

function WellDetail() {
  const { wellId } = useParams()
  const navigate = useNavigate()
  const [well, setWell] = useState(null)
  const [spatial, setSpatial] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      getWellDetail(wellId),
      predictSpatialImpact(wellId)
    ]).then(([wellRes, spatialRes]) => {
      setWell(wellRes.data)
      setSpatial(spatialRes.data)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [wellId])

  if (loading) return <div className="p-6 text-oil-accent animate-pulse">Loading well data...</div>
  if (!well) return <div className="p-6 text-red-400">Well not found</div>

  const history = well.production_history || []
  const timestamps = history.map(h => h.timestamp)
  const production = history.map(h => h.production_rate)
  const pressure = history.map(h => h.reservoir_pressure)

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="p-4 space-y-4">
      <div className="flex items-center gap-4">
        <button onClick={() => navigate('/')} className="text-gray-400 hover:text-white text-sm">← Back</button>
        <h1 className="text-xl font-bold">{wellId}</h1>
        <span className="bg-oil-card border border-oil-border rounded px-2 py-1 text-xs text-oil-accent">
          {well.reservoir_zone}
        </span>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        {[
          { label: 'Depth', value: `${Math.round(well.depth)} ft` },
          { label: 'Permeability', value: well.permeability_index?.toFixed(1) },
          { label: 'Porosity', value: `${(well.porosity_index * 100)?.toFixed(1)}%` },
          { label: 'Dist to Fault', value: well.distance_to_fault_line?.toFixed(4) },
          { label: 'Zone', value: well.reservoir_zone },
        ].map(s => (
          <div key={s.label} className="bg-oil-card border border-oil-border rounded-lg p-3">
            <p className="text-xs text-gray-400">{s.label}</p>
            <p className="text-sm font-bold text-white">{s.value}</p>
          </div>
        ))}
      </div>

      {history.length > 0 && (
        <div className="bg-oil-card border border-oil-border rounded-xl p-4">
          <h2 className="text-sm font-semibold mb-2 text-gray-300">Production History</h2>
          <Plot
            data={[
              { x: timestamps, y: production, type: 'scatter', mode: 'lines', name: 'Production (bbl/d)', line: { color: '#00d4aa', width: 1.5 } },
              { x: timestamps, y: pressure, type: 'scatter', mode: 'lines', name: 'Pressure (psi)', yaxis: 'y2', line: { color: '#3b82f6', width: 1.5 } }
            ]}
            layout={{
              paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
              font: { color: '#94a3b8', size: 10 },
              margin: { t: 20, b: 40, l: 50, r: 50 },
              height: 280,
              xaxis: { gridcolor: '#1e293b' },
              yaxis: { title: 'Production (bbl/d)', gridcolor: '#1e293b', titlefont: { color: '#00d4aa' } },
              yaxis2: { title: 'Pressure (psi)', overlaying: 'y', side: 'right', titlefont: { color: '#3b82f6' } },
              legend: { orientation: 'h', y: -0.15 },
              showlegend: true
            }}
            config={{ responsive: true, displayModeBar: false }}
            className="w-full"
          />
        </div>
      )}

      {spatial && (
        <div className="bg-oil-card border border-oil-border rounded-xl p-4">
          <h2 className="text-sm font-semibold mb-3 text-gray-300">
            Spatial Influence — {spatial.num_neighbors} Connected Neighbors
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-400 border-b border-oil-border">
                  <th className="py-2 text-left">Neighbor</th>
                  <th className="py-2 text-right">Distance</th>
                  <th className="py-2 text-right">Connectivity</th>
                  <th className="py-2 text-right">Production</th>
                  <th className="py-2 text-right">Impact</th>
                  <th className="py-2 text-center">Same Zone</th>
                </tr>
              </thead>
              <tbody>
                {spatial.neighbors?.slice(0, 8).map(n => (
                  <tr key={n.well_id} className="border-b border-oil-border/50 hover:bg-oil-surface cursor-pointer"
                      onClick={() => navigate(`/well/${n.well_id}`)}>
                    <td className="py-2 text-oil-accent">{n.well_id}</td>
                    <td className="py-2 text-right">{n.distance.toFixed(4)}</td>
                    <td className="py-2 text-right">{n.connectivity.toFixed(3)}</td>
                    <td className="py-2 text-right">{Math.round(n.production_rate)} bbl/d</td>
                    <td className="py-2 text-right text-oil-warning">{n.estimated_impact.toFixed(2)}</td>
                    <td className="py-2 text-center">{n.same_zone ? '✓' : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-3 text-xs text-gray-400">
            Total Spatial Impact: <span className="text-oil-accent font-bold">{spatial.total_spatial_impact?.toFixed(2)}</span>
          </div>
        </div>
      )}
    </motion.div>
  )
}

export default WellDetail
