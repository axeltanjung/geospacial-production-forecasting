import { useState, useEffect } from 'react'
import { getHealth } from '../api/client'

export function useApiHealth() {
  const [healthy, setHealthy] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getHealth()
      .then(() => setHealthy(true))
      .catch(() => setHealthy(false))
      .finally(() => setLoading(false))
  }, [])

  return { healthy, loading }
}

export function useInterval(callback, delay) {
  useEffect(() => {
    if (delay === null) return
    const id = setInterval(callback, delay)
    return () => clearInterval(id)
  }, [callback, delay])
}
