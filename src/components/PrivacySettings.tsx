'use client'

import { useState, useEffect } from 'react'
import { Shield, Trash2, Eye, EyeOff, FileText } from 'lucide-react'
import {
  loadUserPreferences,
  setPrivacyMode,
  clearAllUserData,
  getUserStatistics,
} from '@/lib/userPreferences'

interface PrivacySettingsProps {
  onPrivacyModeChange?: (enabled: boolean) => void
}

export default function PrivacySettings({ onPrivacyModeChange }: PrivacySettingsProps) {
  const [privacyMode, setPrivacyModeState] = useState(false)
  const [showConfirmClear, setShowConfirmClear] = useState(false)
  const [stats, setStats] = useState({
    totalRoutes: 0,
    coolRoutesCount: 0,
    coolRoutePercentage: 0,
  })

  useEffect(() => {
    const prefs = loadUserPreferences()
    setPrivacyModeState(prefs.privacyMode)
    setStats(getUserStatistics())
  }, [])

  const togglePrivacyMode = () => {
    const newValue = !privacyMode
    setPrivacyMode(newValue)
    setPrivacyModeState(newValue)
    onPrivacyModeChange?.(newValue)

    if (newValue) {
      setStats({ totalRoutes: 0, coolRoutesCount: 0, coolRoutePercentage: 0 })
    }
  }

  const handleClearData = () => {
    clearAllUserData()
    setStats({ totalRoutes: 0, coolRoutesCount: 0, coolRoutePercentage: 0 })
    setShowConfirmClear(false)
    alert('All your data has been cleared.')
  }

  return (
    <div
      className="bg-white rounded-lg shadow-sm border border-gray-300 p-4"
      role="region"
      aria-label="Privacy Settings"
    >
      <div className="flex items-center gap-2 mb-3">
        <Shield className="w-5 h-5 text-blue-700" aria-hidden="true" />
        <h3 className="font-bold text-gray-900">Privacy & Data</h3>
      </div>

      <div className="space-y-4">
        {/* Anonymous Mode Toggle */}
        <div className="p-3 bg-gray-100 rounded-lg border border-gray-300">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2 text-gray-800">
              {privacyMode ? (
                <EyeOff className="w-4 h-4" aria-hidden="true" />
              ) : (
                <Eye className="w-4 h-4" aria-hidden="true" />
              )}
              <span className="text-sm font-semibold">Anonymous Mode</span>
            </div>
            <button
              onClick={togglePrivacyMode}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                privacyMode ? 'bg-blue-600' : 'bg-gray-300'
              }`}
              role="switch"
              aria-checked={privacyMode}
              aria-label="Toggle anonymous mode"
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  privacyMode ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          <p className="text-xs text-gray-700">
            When enabled, no usage data or preferences are saved.
          </p>
        </div>

        {/* Usage Statistics */}
        {!privacyMode && stats.totalRoutes > 0 && (
          <div className="p-3 bg-green-100 rounded-lg border border-green-300">
            <p className="text-xs font-semibold text-gray-800 mb-2">Your Impact</p>
            <div className="space-y-1 text-xs text-gray-700">
              <div className="flex justify-between">
                <span>Total routes planned:</span>
                <span className="font-semibold">{stats.totalRoutes}</span>
              </div>
              <div className="flex justify-between">
                <span>Cool routes chosen:</span>
                <span className="font-semibold text-green-700">
                  {stats.coolRoutesCount} ({stats.coolRoutePercentage.toFixed(0)}%)
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Privacy Policy Link */}
        <a
          href="/privacy-policy"
          className="flex items-center gap-2 text-sm text-blue-700 hover:text-blue-900 transition-colors focus:outline-none focus:underline"
          aria-label="Read privacy policy"
        >
          <FileText className="w-4 h-4" aria-hidden="true" />
          <span>Privacy Policy</span>
        </a>

        {/* Clear All Data Button */}
        {!privacyMode && (
          <div>
            {!showConfirmClear ? (
              <button
                onClick={() => setShowConfirmClear(true)}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-red-100 hover:bg-red-200 text-red-700 rounded-lg transition-colors text-sm font-medium focus:outline-none focus:ring-2 focus:ring-red-500 border border-red-300"
                aria-label="Clear all stored data"
              >
                <Trash2 className="w-4 h-4" aria-hidden="true" />
                <span>Clear All Data</span>
              </button>
            ) : (
              <div className="p-3 bg-red-100 border border-red-300 rounded-lg space-y-2">
                <p className="text-sm text-red-900 font-medium">
                  Are you sure? This cannot be undone.
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={handleClearData}
                    className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-red-500"
                    aria-label="Confirm clear all data"
                  >
                    Yes, Clear All
                  </button>
                  <button
                    onClick={() => setShowConfirmClear(false)}
                    className="flex-1 px-3 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500"
                    aria-label="Cancel clear data"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Data Collection Notice */}
        <div className="text-xs text-gray-700 pt-2 border-t border-gray-300">
          <p>
            <strong>What we collect:</strong> Route preferences, frequent locations
            (stored locally only). <strong>No server-side tracking.</strong>
          </p>
        </div>
      </div>
    </div>
  )
}
