'use client'

import { useEffect, useRef, memo } from 'react'
import L from 'leaflet'
import type { Feature, Geometry } from 'geojson'
import 'leaflet/dist/leaflet.css'
import type { HexagonFeatureCollection, HexagonProperties, RouteComparison } from '@/lib/api'

// Fix for default marker icons in Leaflet
if (typeof window !== 'undefined') {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  delete (L.Icon.Default.prototype as unknown as Record<string, unknown>)._getIconUrl
  L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
    iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
    shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  })
}

type RouteFeature = RouteComparison['cool_route'] | RouteComparison['fast_route']

interface MapProps {
  centerLat: number
  centerLon: number
  zoom?: number
  hexagons?: HexagonFeatureCollection
  routes?: {
    fast?: RouteFeature
    cool?: RouteFeature
  }
  startPoint?: [number, number] | null
  endPoint?: [number, number] | null
  onMapClick?: (lat: number, lon: number) => void
  showHeatmap?: boolean
}

function MapComponent({
  centerLat,
  centerLon,
  zoom = 14,
  hexagons,
  routes,
  startPoint,
  endPoint,
  onMapClick,
  showHeatmap = true,
}: MapProps) {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<L.Map | null>(null)
  const markersRef = useRef<L.Marker[]>([])
  const layersRef = useRef<{ [key: string]: L.Layer }>({})
  const onMapClickRef = useRef(onMapClick)

  // Keep the ref updated with the latest callback
  useEffect(() => {
    onMapClickRef.current = onMapClick
  }, [onMapClick])

  // Initialize map once
  useEffect(() => {
    if (!mapContainer.current || map.current) return

    // Create map
    map.current = L.map(mapContainer.current, {
      center: [centerLat, centerLon],
      zoom: zoom,
    })

    // Add tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map.current)

    // Handle map clicks using ref
    map.current.on('click', (e: L.LeafletMouseEvent) => {
      if (onMapClickRef.current) {
        onMapClickRef.current(e.latlng.lat, e.latlng.lng)
      }
    })

    return () => {
      if (map.current) {
        map.current.remove()
        map.current = null
      }
    }
  }, []) // Only run once on mount

  // Update center when city changes
  useEffect(() => {
    if (map.current) {
      map.current.flyTo([centerLat, centerLon], zoom, { duration: 1 })
    }
  }, [centerLat, centerLon, zoom])

  // Update hexagon heatmap
  useEffect(() => {
    if (!map.current || !hexagons || !showHeatmap) {
      // Remove existing hexagon layers if they exist
      if (layersRef.current['hexagons']) {
        map.current?.removeLayer(layersRef.current['hexagons'])
        delete layersRef.current['hexagons']
      }
      if (layersRef.current['hexagons-outline']) {
        map.current?.removeLayer(layersRef.current['hexagons-outline'])
        delete layersRef.current['hexagons-outline']
      }
      return
    }

    // Remove existing layers
    if (layersRef.current['hexagons']) {
      map.current.removeLayer(layersRef.current['hexagons'])
    }
    if (layersRef.current['hexagons-outline']) {
      map.current.removeLayer(layersRef.current['hexagons-outline'])
    }

    // Style function
    const style = (feature: Feature<Geometry, HexagonProperties> | undefined) => {
      const comfortScore = feature?.properties?.comfort_score || 0
      let color = '#ef4444'

      if (comfortScore >= 0.7) color = '#22c55e'
      else if (comfortScore >= 0.5) color = '#84cc16'
      else if (comfortScore >= 0.3) color = '#eab308'
      else if (comfortScore >= 0) color = '#f97316'

      return {
        fillColor: color,
        fillOpacity: 0.5,
        color: '#666',
        weight: 0.5,
        opacity: 0.3,
      }
    }

    // Create a popup for hover
    const popup = L.popup({
      closeButton: false,
      closeOnClick: false,
    })

    // Add fill layer
    const hexagonLayer = L.geoJSON(hexagons, {
      style: style,
      onEachFeature: (feature, layer) => {
        const props = feature.properties

        // Show popup on hover
        layer.on('mouseover', (e) => {
          const html = `
            <div class="text-sm">
              <div class="font-semibold mb-1">Thermal Comfort</div>
              <div>Score: ${(props.comfort_score * 100).toFixed(0)}%</div>
              <div>Category: ${props.category}</div>
              <div class="mt-2 text-xs text-gray-600">
                <div>NDVI: ${props.ndvi.toFixed(2)}</div>
                <div>LST: ${props.lst.toFixed(1)}°C</div>
              </div>
            </div>
          `
          popup.setLatLng(e.latlng).setContent(html).openOn(map.current!)
        })

        // Hide popup on mouseout
        layer.on('mouseout', () => {
          popup.remove()
        })

        // Make clicks pass through to the map
        layer.on('click', (e) => {
          if (onMapClickRef.current) {
            onMapClickRef.current(e.latlng.lat, e.latlng.lng)
          }
        })
      },
    }).addTo(map.current)

    layersRef.current['hexagons'] = hexagonLayer
  }, [hexagons, showHeatmap])

  // Update routes
  useEffect(() => {
    if (!map.current) return

    // Remove existing route layers
    if (layersRef.current['cool-route']) {
      map.current.removeLayer(layersRef.current['cool-route'])
      delete layersRef.current['cool-route']
    }
    if (layersRef.current['fast-route']) {
      map.current.removeLayer(layersRef.current['fast-route'])
      delete layersRef.current['fast-route']
    }

    if (!routes) return

    // Add cool route
    if (routes.cool) {
      const coolLayer = L.geoJSON(routes.cool, {
        style: {
          color: '#22c55e',
          weight: 5,
          opacity: 0.8,
        },
      }).addTo(map.current)
      layersRef.current['cool-route'] = coolLayer
    }

    // Add fast route
    if (routes.fast) {
      const fastLayer = L.geoJSON(routes.fast, {
        style: {
          color: '#ef4444',
          weight: 5,
          opacity: 0.8,
        },
      }).addTo(map.current)
      layersRef.current['fast-route'] = fastLayer
    }

    // Fit bounds to routes
    const route = routes.cool || routes.fast
    if (route) {
      const coordinates =
        route.geometry.type === 'MultiLineString'
          ? (route.geometry.coordinates as number[][][]).flat()
          : (route.geometry.coordinates as number[][])

      if (coordinates.length > 0) {
        const bounds = L.latLngBounds(
          coordinates.map((coord) => [coord[1], coord[0]] as [number, number])
        )
        map.current.fitBounds(bounds, { padding: [100, 100], duration: 1 })
      }
    }
  }, [routes])

  // Update markers
  useEffect(() => {
    if (!map.current) return

    // Clear existing markers
    markersRef.current.forEach((marker) => marker.remove())
    markersRef.current = []

    // Create marker icon
    const createIcon = (color: string) =>
      L.divIcon({
        className: 'custom-marker',
        html: `<div class="w-8 h-8 ${color} rounded-full border-4 border-white shadow-lg"></div>`,
        iconSize: [32, 32],
        iconAnchor: [16, 16],
      })

    // Add start marker
    if (startPoint) {
      const marker = L.marker([startPoint[0], startPoint[1]], {
        icon: createIcon('bg-green-500'),
      })
        .bindPopup('<div class="font-semibold">Start</div>')
        .addTo(map.current)
      markersRef.current.push(marker)
    }

    // Add end marker
    if (endPoint) {
      const marker = L.marker([endPoint[0], endPoint[1]], {
        icon: createIcon('bg-red-500'),
      })
        .bindPopup('<div class="font-semibold">End</div>')
        .addTo(map.current)
      markersRef.current.push(marker)
    }
  }, [startPoint, endPoint])

  return (
    <div className="relative w-full h-full">
      <div ref={mapContainer} className="w-full h-full rounded-lg overflow-hidden" />
    </div>
  )
}

// Memo the component to prevent unnecessary re-renders with React Compiler
export default memo(MapComponent)
