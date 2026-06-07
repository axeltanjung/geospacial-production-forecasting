import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
})

export const getHealth = () => api.get('/health')
export const getMapData = () => api.get('/map/data')
export const getWellDetail = (wellId) => api.get(`/well/${wellId}`)
export const predictProduction = (wellId, horizon = 30) =>
  api.post('/predict/production', { well_id: wellId, horizon })
export const predictSpatialImpact = (wellId) =>
  api.post('/predict/spatial-impact', { well_id: wellId })
export const getGraphStructure = () => api.get('/graph/structure')
export const multiStepForecast = (wellId) =>
  api.post('/forecast/multi-step', { well_id: wellId })
export const explainSpatial = (wellId) =>
  api.post('/explain/spatial', { well_id: wellId })
export const listWells = () => api.get('/wells')
export const listZones = () => api.get('/zones')

export default api
