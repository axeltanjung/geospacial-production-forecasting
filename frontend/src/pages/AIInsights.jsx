import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Plot from 'react-plotly.js'
import { listWells, explainSpatial } from '../api/client'

function AIInsights() {
  const [wells, setWells] = useState([])
  const [selectedWell, setSelectedWell] = useState('')
  const [explanation, setExplanation] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    listWells().then(res => {
      setWells(res.data.wells || [])
      if (res.data.wells?.length > 0) setSelectedWell(res.data.wells[0].well_id)
    })
  }, [])

  useEffect(() => {
    if (!selectedWell) return
    setLoading(true)
    explainSpatial(selectedWell)
      .then(res => { setExplanation(res.data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [selectedWell])

  const neighborChart = () => {
    if (!explanation?.neighbors) return null
    const neighbors = explanation.neighbors.slice(0, 8)
    return (
      <Plot
        data={[{
          x: neighbors.map(n => n.well_id),
          y: neighbors.map(n => n.connectivity),
          type: 'bar',
          marker: { color: neighbors.map(n => n.same_zone ? '#00d4aa' : '#3b82f6') }
        }]}
        layout={{
          paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
          font: { color: '#94a3b8', size: 10 },
          margin: { t: 10, b: 60, l: 50, r: 20 },
          height: 250,
          xaxis: { title: 'Neighbor Well', gridcolor: '#1e293b' },
          yaxis: { title: 'Connectivity', gridcolor: '#1e293b' },
        }}
        config={{ responsive: true, displayModeBar: false }}
        className="w-full"
      />
    )
  }

  const impactChart = () => {
    if (!explanation?.neighbors) return null
    const neighbors = explanation.neighbors.slice(0, 8)
    return (
      <Plot
        data={[{
          x: neighbors.map(n => n.distance),
          y: neighbors.map(n => n.estimated_impact),
          text: neighbors.map(n => n.well_id),
          mode: 'markers+text',
          type: 'scatter',
          textposition: 'top center',
          textfont: { size: 9, color: '#94a3b8' },
          marker: { size: neighbors.map(n => Math.max(8, n.production_rate / 50)), color: '#8b5cf6', opacity: 0.7 }
        }]}
        layout={{
          paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
          font: { color: '#94a3b8', size: 10 },
          margin: { t: 10, b: 50, l: 60, r: 20 },
          height: 250,
          xaxis: { title: 'Distance', gridcolor: '#1e293b' },
          yaxis: { title: 'Estimated Impact', gridcolor: '#1e293b' },
        }}
        config={{ responsive: true, displayModeBar: false }}
        className="w-full"
      />
    )
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">AI Spatial Insights</h1>
          <p className="text-sm text-gray-400">Explainable spatial dependency analysis & neighbor influence</p>
        </div>
        <select
          value={selectedWell}
          onChange={e => setSelectedWell(e.target.value)}
          className="bg-oil-card border border-oil-border rounded-lg px-3 py-2 text-sm text-white"
        >
          {wells.map(w => <option key={w.well_id} value={w.well_id}>{w.well_id}</option>)}
        </select>
      </div>

      {loading && <div className="text-oil-accent animate-pulse">Analyzing spatial dependencies...</div>}

      {explanation && (
        <>
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-oil-card border border-oil-border rounded-lg p-4">
              <p className="text-xs text-gray-400">Connected Neighbors</p>
              <p className="text-2xl font-bold text-oil-accent">{explanation.neighbors?.length || 0}</p>
            </div>
            <div className="bg-oil-card border border-oil-border rounded-lg p-4">
              <p className="text-xs text-gray-400">Total Spatial Impact</p>
              <p className="text-2xl font-bold text-oil-purple">{explanation.total_spatial_impact?.toFixed(2)}</p>
            </div>
            <div className="bg-oil-card border border-oil-border rounded-lg p-4">
              <p className="text-xs text-gray-400">Reservoir Zone</p>
              <p className="text-2xl font-bold text-oil-blue">{explanation.zone}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-oil-card border border-oil-border rounded-xl p-4">
              <h3 className="text-sm font-semibold mb-2 text-gray-300">Neighbor Connectivity Strength</h3>
              {neighborChart()}
              <p className="text-xs text-gray-500 mt-1">Green = same zone | Blue = different zone</p>
            </div>
            <div className="bg-oil-card border border-oil-border rounded-xl p-4">
              <h3 className="text-sm font-semibold mb-2 text-gray-300">Distance vs Impact (bubble = production)</h3>
              {impactChart()}
            </div>
          </div>

          <div className="bg-oil-card border border-oil-border rounded-xl p-4">
            <h3 className="text-sm font-semibold mb-3 text-gray-300">Spatial Dependency Explanation</h3>
            <div className="space-y-2 text-sm text-gray-300">
              <p>Well <span className="text-oil-accent font-bold">{selectedWell}</span> is connected to <span className="text-white font-bold">{explanation.neighbors?.length}</span> neighbors in <span className="text-oil-blue font-bold">{explanation.zone}</span>.</p>
              <p>The strongest spatial influence comes from wells in the same reservoir zone, where pressure communication through shared rock matrix creates production correlation.</p>
              {explanation.neighbors?.[0] && (
                <p>Closest neighbor <span className="text-oil-accent">{explanation.neighbors[0].well_id}</span> has connectivity weight <span className="text-white font-bold">{explanation.neighbors[0].connectivity.toFixed(3)}</span> — indicating strong reservoir coupling.</p>
              )}
              <p className="text-gray-500 text-xs mt-3">This analysis uses the spatial adjacency matrix weighted by distance decay and fault-line barriers to quantify well-to-well influence.</p>
            </div>
          </div>
        </>
      )}
    </motion.div>
  )
}

export default AIInsights
