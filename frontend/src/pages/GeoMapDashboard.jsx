import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { MapContainer, TileLayer, CircleMarker, Polyline, Popup, useMap } from 'react-leaflet'
import { getMapData } from '../api/client'
import 'leaflet/dist/leaflet.css'

function getProductionColor(rate) {
  if (rate > 1000) return '#00d4aa'
  if (rate > 500) return '#3b82f6'
  if (rate > 200) return '#f59e0b'
  return '#ef4444'
}

function getProductionRadius(rate) {
  return Math.max(4, Math.min(12, rate / 150))
}

function MapBoundsUpdater({ bounds }) {
  const map = useMap()
  useEffect(() => {
    if (bounds) {
      map.fitBounds([[bounds.lat_min, bounds.lon_min], [bounds.lat_max, bounds.lon_max]], { padding: [20, 20] })
    }
  }, [bounds, map])
  return null
}

function StatsPanel({ data }) {
  if (!data.wells.length) return null
  const avgProd = data.wells.reduce((s, w) => s + w.production_rate, 0) / data.wells.length
  const totalProd = data.wells.reduce((s, w) => s + w.production_rate, 0)
  const zones = [...new Set(data.wells.map(w => w.zone))]
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
      {[
        { label: 'Total Wells', value: data.wells.length, color: 'text-oil-accent' },
        { label: 'Total Production', value: `${Math.round(totalProd).toLocaleString()} bbl/d`, color: 'text-oil-blue' },
        { label: 'Avg Rate', value: `${Math.round(avgProd)} bbl/d`, color: 'text-oil-purple' },
        { label: 'Zones', value: zones.length, color: 'text-oil-warning' },
      ].map((stat, i) => (
        <motion.div
          key={stat.label}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.1 }}
          className="bg-oil-card border border-oil-border rounded-lg p-3"
        >
          <p className="text-xs text-gray-400">{stat.label}</p>
          <p className={`text-lg font-bold ${stat.color}`}>{stat.value}</p>
        </motion.div>
      ))}
    </div>
  )
}

function GeoMapDashboard() {
  const [mapData, setMapData] = useState({ wells: [], faults: [], field_bounds: null })
  const [loading, setLoading] = useState(true)
  const [selectedZone, setSelectedZone] = useState('all')
  const navigate = useNavigate()

  useEffect(() => {
    getMapData()
      .then(res => {
        setMapData(res.data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const filteredWells = selectedZone === 'all'
    ? mapData.wells
    : mapData.wells.filter(w => w.zone === selectedZone)

  const zones = [...new Set(mapData.wells.map(w => w.zone))]

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="p-4 h-full flex flex-col"
    >
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-xl font-bold text-white">Geospatial Production Map</h1>
          <p className="text-sm text-gray-400">Real-time well production visualization with reservoir zones</p>
        </div>
        <select
          value={selectedZone}
          onChange={e => setSelectedZone(e.target.value)}
          className="bg-oil-card border border-oil-border rounded-lg px-3 py-2 text-sm text-white"
        >
          <option value="all">All Zones</option>
          {zones.map(z => <option key={z} value={z}>{z}</option>)}
        </select>
      </div>

      <StatsPanel data={{ wells: filteredWells }} />

      <div className="flex-1 rounded-xl overflow-hidden border border-oil-border min-h-[400px]">
        {loading ? (
          <div className="h-full flex items-center justify-center bg-oil-card">
            <div className="text-oil-accent animate-pulse">Loading map data...</div>
          </div>
        ) : (
          <MapContainer
            center={[31.5, -103.5]}
            zoom={10}
            className="h-full w-full"
            style={{ background: '#0a0e1a' }}
          >
            <TileLayer
              url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
              attribution='&copy; <a href="https://carto.com/">CARTO</a>'
            />
            {mapData.field_bounds && <MapBoundsUpdater bounds={mapData.field_bounds} />}

            {mapData.faults.map((fault, i) => (
              <Polyline
                key={`fault-${i}`}
                positions={[[fault.start_lat, fault.start_lon], [fault.end_lat, fault.end_lon]]}
                color="#ef4444"
                weight={2}
                opacity={0.6}
                dashArray="5,5"
              />
            ))}

            {filteredWells.map(well => (
              <CircleMarker
                key={well.well_id}
                center={[well.latitude, well.longitude]}
                radius={getProductionRadius(well.production_rate)}
                fillColor={getProductionColor(well.production_rate)}
                fillOpacity={0.8}
                color={getProductionColor(well.production_rate)}
                weight={1}
                eventHandlers={{
                  click: () => navigate(`/well/${well.well_id}`)
                }}
              >
                <Popup>
                  <div className="text-sm">
                    <p className="font-bold">{well.well_id}</p>
                    <p>Rate: {Math.round(well.production_rate)} bbl/d</p>
                    <p>Pressure: {Math.round(well.reservoir_pressure)} psi</p>
                    <p>Zone: {well.zone}</p>
                    <p>Depth: {Math.round(well.depth)} ft</p>
                  </div>
                </Popup>
              </CircleMarker>
            ))}
          </MapContainer>
        )}
      </div>

      <div className="mt-3 flex items-center gap-6 text-xs text-gray-400">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#00d4aa] inline-block"></span> &gt;1000 bbl/d</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#3b82f6] inline-block"></span> 500-1000</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#f59e0b] inline-block"></span> 200-500</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-[#ef4444] inline-block"></span> &lt;200</span>
        <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-red-500 inline-block" style={{borderTop: '2px dashed'}}></span> Fault Line</span>
      </div>
    </motion.div>
  )
}

export default GeoMapDashboard
