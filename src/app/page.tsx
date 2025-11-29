'use client'

import { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { api, City, RouteComparison, CityData, ProgressEvent } from '@/lib/api'
import RouteCard from '@/components/RouteCard'
import ProgressDialog from '@/components/ProgressDialog'
import { MapPin, Navigation, Info, Loader2 } from 'lucide-react'

const Map = dynamic(() => import('@/components/Map'), { ssr: false })

type SelectionMode = 'start' | 'end'

export default function Home() {
  const [cities, setCities] = useState<City[]>([])
  const [selectedCity, setSelectedCity] = useState<City | null>(null)
  const [cityData, setCityData] = useState<CityData | null>(null)
  const [loadingCity, setLoadingCity] = useState(false)
  const [selectionMode, setSelectionMode] = useState<SelectionMode>('start')
  const [startPoint, setStartPoint] = useState<[number, number] | null>(null)
  const [endPoint, setEndPoint] = useState<[number, number] | null>(null)
  const [routes, setRoutes] = useState<RouteComparison | null>(null)
  const [loadingRoutes, setLoadingRoutes] = useState(false)
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [showBothRoutes, setShowBothRoutes] = useState(true)
  const [progressEvents, setProgressEvents] = useState<ProgressEvent[]>([])
  const [showProgress, setShowProgress] = useState(false)

  useEffect(() => {
    loadCities()
  }, [])

  async function loadCities() {
    try {
      const citiesList = await api.getCities()
      setCities(citiesList)
      if (citiesList.length > 0) setSelectedCity(citiesList[0])
    } catch (error) {
      console.error('Failed to load cities:', error)
    }
  }

  useEffect(() => {
    if (selectedCity) loadCityDataIfNeeded()
  }, [selectedCity])

  async function loadCityDataIfNeeded() {
    if (!selectedCity) return
    setLoadingCity(true)
    try {
      const data = await api.getCityData(selectedCity.name)
      setCityData(data)
      setLoadingCity(false)
    } catch (error: any) {
      if (error.message.includes('not loaded yet')) {
        await initiateLoadingWithProgress()
      } else {
        console.error('Failed to load city data:', error)
        setLoadingCity(false)
      }
    }
  }

  async function initiateLoadingWithProgress() {
    if (!selectedCity) return
    try {
      setProgressEvents([])
      setShowProgress(true)
      const operationId = await api.loadCityData(selectedCity.name)
      api.subscribeToProgress(
        operationId,
        (event) => setProgressEvents((prev) => [...prev, event]),
        async () => {
          setProgressEvents((prev) => [...prev, { message: 'Complete!', complete: true, timestamp: new Date().toISOString() }])
          setTimeout(async () => {
            const cityData = await api.getCityData(selectedCity.name)
            setCityData(cityData)
            setLoadingCity(false)
          }, 1000)
        },
        (error) => {
          setProgressEvents((prev) => [...prev, { message: `Error: ${error}`, error: true, timestamp: new Date().toISOString() }])
          setLoadingCity(false)
        }
      )
    } catch (error) {
      setLoadingCity(false)
      setShowProgress(false)
    }
  }

  function handleMapClick(lat: number, lon: number) {
    if (selectionMode === 'start') setStartPoint([lat, lon])
    else setEndPoint([lat, lon])
  }

  async function findRoutes() {
    if (!selectedCity || !startPoint || !endPoint) return
    setLoadingRoutes(true)
    setRoutes(null)
    try {
      const comparison = await api.compareRoutes(selectedCity.name, startPoint[0], startPoint[1], endPoint[0], endPoint[1])
      setRoutes(comparison)
    } catch (error) {
      alert('Could not find routes.')
    } finally {
      setLoadingRoutes(false)
    }
  }

  function clearPoints() {
    setStartPoint(null)
    setEndPoint(null)
    setRoutes(null)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-green-600 rounded-lg flex items-center justify-center">
              <MapPin className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">GreenPath</h1>
              <p className="text-sm text-gray-600">Shade & Walkability Navigator</p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="bg-white rounded-lg shadow-sm border p-4 mb-6">
          <div className="flex flex-wrap items-center gap-4">
            <label className="text-sm font-semibold text-gray-700">City:</label>
            <div className="flex gap-2">
              {cities.map((city) => (
                <button key={city.name} onClick={() => { setSelectedCity(city); setCityData(null); clearPoints(); }} className={`px-4 py-2 rounded-lg font-medium ${selectedCity?.name === city.name ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}>
                  {city.name}
                </button>
              ))}
            </div>
          </div>
        </div>

        {!cityData && !loadingCity && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6 flex items-start gap-3">
            <Info className="w-5 h-5 text-blue-600 mt-0.5" />
            <div className="text-sm text-blue-800">
              <p className="font-semibold mb-1">Data not loaded</p>
              <p>Click city button again to load satellite data (takes 2-5 min first time).</p>
            </div>
          </div>
        )}

        {routes && (
          <div className="bg-white rounded-lg shadow-sm border p-6 mb-6">
            <h2 className="text-xl font-bold mb-4">Route Comparison</h2>
            <div className="grid md:grid-cols-3 gap-6">
              <RouteCard title="Cool Route" stats={routes.cool_route.properties} color="green" />
              <div className="flex flex-col justify-center space-y-3 text-sm">
                <div className="text-center p-3 bg-gray-50 rounded">
                  <div className="text-gray-600">Distance Diff</div>
                  <div className="text-lg font-bold">+{routes.comparison.distance_diff_pct.toFixed(1)}%</div>
                </div>
                <div className="text-center p-3 bg-green-50 rounded">
                  <div className="text-gray-600">Comfort +</div>
                  <div className="text-lg font-bold text-green-600">+{(routes.comparison.comfort_improvement * 100).toFixed(1)}%</div>
                </div>
              </div>
              <RouteCard title="Fast Route" stats={routes.fast_route.properties} color="red" />
            </div>
          </div>
        )}

        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-4">
            <div className="bg-white rounded-lg shadow-sm border p-4">
              <h3 className="font-bold mb-3">Select Points</h3>
              <div className="space-y-3">
                <div className="flex gap-2">
                  <button onClick={() => setSelectionMode('start')} className={`flex-1 px-4 py-2 rounded-lg ${selectionMode === 'start' ? 'bg-green-600 text-white' : 'bg-gray-100 text-gray-700'}`}>Start</button>
                  <button onClick={() => setSelectionMode('end')} className={`flex-1 px-4 py-2 rounded-lg ${selectionMode === 'end' ? 'bg-red-600 text-white' : 'bg-gray-100 text-gray-700'}`}>End</button>
                </div>
                <button onClick={findRoutes} disabled={!startPoint || !endPoint || loadingRoutes || !cityData} className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white font-semibold py-3 px-4 rounded-lg flex items-center justify-center gap-2">
                  {loadingRoutes ? <><Loader2 className="w-4 h-4 animate-spin" />Finding...</> : <><Navigation className="w-4 h-4" />Find Routes</>}
                </button>
                <button onClick={clearPoints} className="w-full bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium py-2 px-4 rounded-lg">Clear</button>
              </div>
            </div>
            <div className="bg-white rounded-lg shadow-sm border p-4">
              <h3 className="font-bold mb-3">Options</h3>
              <div className="space-y-2">
                <label className="flex items-center gap-2"><input type="checkbox" checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} className="w-4 h-4" /><span className="text-sm">Show Heatmap</span></label>
                <label className="flex items-center gap-2"><input type="checkbox" checked={showBothRoutes} onChange={(e) => setShowBothRoutes(e.target.checked)} className="w-4 h-4" /><span className="text-sm">Both Routes</span></label>
              </div>
            </div>
          </div>
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-sm border p-4">
              <div className="h-[600px]">
                {selectedCity && <Map centerLat={selectedCity.lat} centerLon={selectedCity.lon} zoom={14} hexagons={cityData?.hexagons} routes={routes ? { cool: routes.cool_route, fast: showBothRoutes ? routes.fast_route : undefined } : undefined} startPoint={startPoint} endPoint={endPoint} onMapClick={handleMapClick} showHeatmap={showHeatmap && !!cityData} />}
              </div>
            </div>
          </div>
        </div>
      </main>

      <ProgressDialog isOpen={showProgress} events={progressEvents} onClose={() => setShowProgress(false)} />
    </div>
  )
}