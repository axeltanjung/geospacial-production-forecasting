import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import GeoMapDashboard from './pages/GeoMapDashboard'
import WellDetail from './pages/WellDetail'
import AIInsights from './pages/AIInsights'
import ForecastPage from './pages/ForecastPage'
import NetworkPage from './pages/NetworkPage'

const navItems = [
  { path: '/', label: 'Geo Map', icon: '🗺️' },
  { path: '/forecast', label: 'Forecast', icon: '📈' },
  { path: '/network', label: 'Network', icon: '🕸️' },
  { path: '/insights', label: 'AI Insights', icon: '🧠' },
]

function Sidebar() {
  const location = useLocation()
  return (
    <aside className="w-16 lg:w-56 bg-oil-surface border-r border-oil-border flex flex-col py-4 shrink-0">
      <div className="px-4 mb-8 hidden lg:block">
        <h1 className="text-oil-accent font-bold text-sm tracking-wider">GEORESERVOIR</h1>
        <p className="text-xs text-gray-500 mt-1">AI Production Platform</p>
      </div>
      <nav className="flex-1 space-y-1 px-2">
        {navItems.map(item => {
          const active = location.pathname === item.path
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all ${
                active
                  ? 'bg-oil-accent/10 text-oil-accent border border-oil-accent/30'
                  : 'text-gray-400 hover:text-white hover:bg-oil-card'
              }`}
            >
              <span className="text-lg">{item.icon}</span>
              <span className="hidden lg:inline">{item.label}</span>
            </Link>
          )
        })}
      </nav>
      <div className="px-4 hidden lg:block">
        <div className="text-xs text-gray-600 border-t border-oil-border pt-3">
          <p>v1.0.0</p>
          <p className="mt-1">Spatio-Temporal AI</p>
        </div>
      </div>
    </aside>
  )
}

function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen overflow-hidden bg-oil-dark">
        <Sidebar />
        <main className="flex-1 overflow-auto">
          <AnimatePresence mode="wait">
            <Routes>
              <Route path="/" element={<GeoMapDashboard />} />
              <Route path="/well/:wellId" element={<WellDetail />} />
              <Route path="/forecast" element={<ForecastPage />} />
              <Route path="/network" element={<NetworkPage />} />
              <Route path="/insights" element={<AIInsights />} />
            </Routes>
          </AnimatePresence>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
