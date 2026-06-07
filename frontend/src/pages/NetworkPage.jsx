import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { getGraphStructure } from '../api/client'

function NetworkPage() {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] })
  const [loading, setLoading] = useState(true)
  const canvasRef = useRef(null)

  useEffect(() => {
    getGraphStructure()
      .then(res => { setGraphData(res.data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!canvasRef.current || !graphData.nodes.length) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const rect = canvas.parentElement.getBoundingClientRect()
    canvas.width = rect.width
    canvas.height = rect.height - 20

    ctx.fillStyle = '#0a0e1a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    const nodes = graphData.nodes
    const lats = nodes.map(n => n.lat)
    const lons = nodes.map(n => n.lon)
    const latMin = Math.min(...lats), latMax = Math.max(...lats)
    const lonMin = Math.min(...lons), lonMax = Math.max(...lons)

    const scaleX = (lon) => ((lon - lonMin) / (lonMax - lonMin + 0.001)) * (canvas.width - 80) + 40
    const scaleY = (lat) => ((latMax - lat) / (latMax - latMin + 0.001)) * (canvas.height - 80) + 40

    const nodePositions = {}
    nodes.forEach(n => {
      nodePositions[n.id] = { x: scaleX(n.lon), y: scaleY(n.lat) }
    })

    ctx.strokeStyle = 'rgba(59, 130, 246, 0.15)'
    ctx.lineWidth = 0.5
    graphData.edges.slice(0, 300).forEach(edge => {
      const src = nodePositions[edge.source]
      const tgt = nodePositions[edge.target]
      if (!src || !tgt) return
      ctx.beginPath()
      ctx.moveTo(src.x, src.y)
      ctx.lineTo(tgt.x, tgt.y)
      ctx.globalAlpha = Math.min(edge.weight * 2, 0.6)
      ctx.stroke()
    })

    ctx.globalAlpha = 1
    const zoneColors = {
      'Zone_1': '#00d4aa', 'Zone_2': '#3b82f6', 'Zone_3': '#f59e0b',
      'Zone_4': '#8b5cf6', 'Zone_5': '#ef4444'
    }

    nodes.forEach(n => {
      const pos = nodePositions[n.id]
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, 4, 0, Math.PI * 2)
      ctx.fillStyle = zoneColors[n.zone] || '#6b7280'
      ctx.fill()
      ctx.strokeStyle = 'rgba(255,255,255,0.3)'
      ctx.lineWidth = 0.5
      ctx.stroke()
    })
  }, [graphData])

  const stats = {
    nodes: graphData.nodes.length,
    edges: graphData.edges.length,
    avgWeight: graphData.edges.length > 0
      ? (graphData.edges.reduce((s, e) => s + e.weight, 0) / graphData.edges.length).toFixed(3)
      : 0,
    zones: [...new Set(graphData.nodes.map(n => n.zone))].length
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="p-4 h-full flex flex-col">
      <div className="mb-4">
        <h1 className="text-xl font-bold text-white">Well Connectivity Network</h1>
        <p className="text-sm text-gray-400">Spatial dependency graph — edges represent reservoir connectivity</p>
      </div>

      <div className="grid grid-cols-4 gap-3 mb-4">
        {[
          { label: 'Nodes', value: stats.nodes, color: 'text-oil-accent' },
          { label: 'Edges', value: stats.edges, color: 'text-oil-blue' },
          { label: 'Avg Weight', value: stats.avgWeight, color: 'text-oil-purple' },
          { label: 'Zones', value: stats.zones, color: 'text-oil-warning' },
        ].map(s => (
          <div key={s.label} className="bg-oil-card border border-oil-border rounded-lg p-3">
            <p className="text-xs text-gray-400">{s.label}</p>
            <p className={`text-lg font-bold ${s.color}`}>{s.value}</p>
          </div>
        ))}
      </div>

      <div className="flex-1 bg-oil-card border border-oil-border rounded-xl overflow-hidden min-h-[400px]">
        {loading ? (
          <div className="h-full flex items-center justify-center">
            <span className="text-oil-accent animate-pulse">Building graph structure...</span>
          </div>
        ) : (
          <canvas ref={canvasRef} className="w-full h-full" />
        )}
      </div>

      <div className="mt-3 flex items-center gap-6 text-xs text-gray-400">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#00d4aa] inline-block"></span> Zone 1</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#3b82f6] inline-block"></span> Zone 2</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#f59e0b] inline-block"></span> Zone 3</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#8b5cf6] inline-block"></span> Zone 4</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#ef4444] inline-block"></span> Zone 5</span>
      </div>
    </motion.div>
  )
}

export default NetworkPage
