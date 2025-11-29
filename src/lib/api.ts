const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'

export interface City {
  name: string
  lat: number
  lon: number
}

export interface RouteStats {
  distance_km: number
  avg_comfort: number
  walking_time_min: number
  tree_coverage_pct: number
}

export interface RouteComparison {
  fast_route: {
    type: 'Feature'
    geometry: {
      type: 'LineString' | 'MultiLineString'
      coordinates: number[][] | number[][][]
    }
    properties: RouteStats
  }
  cool_route: {
    type: 'Feature'
    geometry: {
      type: 'LineString' | 'MultiLineString'
      coordinates: number[][] | number[][][]
    }
    properties: RouteStats
  }
  comparison: {
    distance_diff_m: number
    distance_diff_pct: number
    comfort_improvement: number
    time_diff_min: number
  }
}

export interface CityData {
  city: string
  hexagons: {
    type: 'FeatureCollection'
    features: Array<{
      type: 'Feature'
      geometry: any
      properties: {
        hex_id: string
        comfort_score: number
        ndvi: number
        lst: number
        slope: number
        shadow: number
        category: string
      }
    }>
  }
  stats: {
    total: number
    mean_comfort: number
    min_comfort: number
    max_comfort: number
  }
}

export interface ProgressEvent {
  message: string
  progress?: number
  complete?: boolean
  error?: boolean
  data?: any
  timestamp: string
}

export class ApiService {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  async getCities(): Promise<City[]> {
    const response = await fetch(`${this.baseUrl}/api/cities`)
    if (!response.ok) {
      throw new Error('Failed to fetch cities')
    }
    const data = await response.json()
    return data.cities
  }

  async getCityData(cityName: string): Promise<CityData> {
    const response = await fetch(`${this.baseUrl}/api/city/${cityName}/data`)
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.error || 'Failed to fetch city data')
    }
    return response.json()
  }

  async loadCityData(cityName: string): Promise<string> {
    const response = await fetch(`${this.baseUrl}/api/city/${cityName}/load`, {
      method: 'POST',
    })
    if (!response.ok) {
      throw new Error('Failed to initiate city data loading')
    }
    const data = await response.json()
    return data.operation_id
  }

  subscribeToProgress(
    operationId: string,
    onProgress: (event: ProgressEvent) => void,
    onComplete: (data?: any) => void,
    onError: (error: string) => void
  ): () => void {
    const eventSource = new EventSource(
      `${this.baseUrl}/api/progress/${operationId}`
    )

    eventSource.onmessage = (event) => {
      try {
        const data: ProgressEvent = JSON.parse(event.data)

        if (data.error) {
          onError(data.message)
          eventSource.close()
        } else if (data.complete) {
          onComplete(data.data)
          eventSource.close()
        } else if (!data.keepalive) {
          onProgress(data)
        }
      } catch (error) {
        console.error('Error parsing SSE data:', error)
      }
    }

    eventSource.onerror = (error) => {
      console.error('SSE error:', error)
      onError('Connection error')
      eventSource.close()
    }

    // Return cleanup function
    return () => {
      eventSource.close()
    }
  }

  async compareRoutes(
    city: string,
    startLat: number,
    startLon: number,
    endLat: number,
    endLon: number
  ): Promise<RouteComparison> {
    const response = await fetch(`${this.baseUrl}/api/routes/compare`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        city,
        start_lat: startLat,
        start_lon: startLon,
        end_lat: endLat,
        end_lon: endLon,
      }),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.error || 'Failed to compare routes')
    }

    return response.json()
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/health`)
      return response.ok
    } catch {
      return false
    }
  }
}

export const api = new ApiService()
