'use client'

import { RouteStats } from '@/lib/api'
import { formatDistance, formatTime, formatComfort, getComfortCategory } from '@/lib/utils'
import { Navigation, Clock, Leaf, TreePine } from 'lucide-react'

interface RouteCardProps {
  title: string
  stats: RouteStats
  color: 'green' | 'red'
}

export default function RouteCard({ title, stats, color }: RouteCardProps) {
  const colorClasses = {
    green: 'border-green-500 bg-green-100',
    red: 'border-red-500 bg-red-100',
  }

  const badgeColors = {
    green: 'bg-green-500',
    red: 'bg-red-500',
  }

  return (
    <div className={`border-2 ${colorClasses[color]} rounded-lg p-4 space-y-3`}>
      <div className="flex items-center gap-2">
        <div className={`w-3 h-3 rounded-full ${badgeColors[color]}`} />
        <h3 className="font-bold text-lg text-gray-900">{title}</h3>
      </div>

      <div className="space-y-2">
        <div className="flex items-center gap-2 text-sm">
          <Navigation className="w-4 h-4 text-gray-700" />
          <span className="text-gray-700">Distance:</span>
          <span className="font-semibold text-gray-900">{formatDistance(stats.distance_km)}</span>
        </div>

        <div className="flex items-center gap-2 text-sm">
          <Clock className="w-4 h-4 text-gray-700" />
          <span className="text-gray-700">Walking Time:</span>
          <span className="font-semibold text-gray-900">{formatTime(stats.walking_time_min)}</span>
        </div>

        <div className="flex items-center gap-2 text-sm">
          <Leaf className="w-4 h-4 text-gray-700" />
          <span className="text-gray-700">Comfort:</span>
          <span className="font-semibold text-gray-900">
            {formatComfort(stats.avg_comfort)} ({getComfortCategory(stats.avg_comfort)})
          </span>
        </div>

        <div className="flex items-center gap-2 text-sm">
          <TreePine className="w-4 h-4 text-gray-700" />
          <span className="text-gray-700">Tree Coverage:</span>
          <span className="font-semibold text-gray-900">{stats.tree_coverage_pct.toFixed(0)}%</span>
        </div>
      </div>
    </div>
  )
}
