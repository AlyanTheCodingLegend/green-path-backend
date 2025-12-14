'use client'

import { useEffect, useState } from 'react'
import { Sun, Moon, Type } from 'lucide-react'
import {
  loadUserPreferences,
  updateAccessibilityPreferences,
} from '@/lib/userPreferences'

interface AccessibilityControlsProps {
  onHighContrastChange?: (enabled: boolean) => void
  onFontSizeChange?: (size: 'normal' | 'large' | 'xlarge') => void
}

export default function AccessibilityControls({
  onHighContrastChange,
  onFontSizeChange,
}: AccessibilityControlsProps) {
  const [highContrast, setHighContrast] = useState(false)
  const [fontSize, setFontSize] = useState<'normal' | 'large' | 'xlarge'>('normal')

  useEffect(() => {
    const prefs = loadUserPreferences()
    setHighContrast(prefs.accessibilityPreferences.highContrast)
    setFontSize(prefs.accessibilityPreferences.fontSize)

    // Apply preferences
    applyAccessibilitySettings(
      prefs.accessibilityPreferences.highContrast,
      prefs.accessibilityPreferences.fontSize
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const applyAccessibilitySettings = (
    contrast: boolean,
    size: 'normal' | 'large' | 'xlarge'
  ) => {
    const root = document.documentElement

    // High contrast mode
    if (contrast) {
      root.classList.add('high-contrast')
    } else {
      root.classList.remove('high-contrast')
    }

    // Font size
    root.classList.remove('font-normal', 'font-large', 'font-xlarge')
    root.classList.add(`font-${size}`)

    onHighContrastChange?.(contrast)
    onFontSizeChange?.(size)
  }

  const toggleHighContrast = () => {
    const newValue = !highContrast
    setHighContrast(newValue)
    updateAccessibilityPreferences(newValue, undefined)
    applyAccessibilitySettings(newValue, fontSize)
  }

  const changeFontSize = (size: 'normal' | 'large' | 'xlarge') => {
    setFontSize(size)
    updateAccessibilityPreferences(undefined, size)
    applyAccessibilitySettings(highContrast, size)
  }

  return (
    <div
      className="bg-white rounded-lg shadow-sm border border-gray-300 p-4"
      role="region"
      aria-label="Accessibility Controls"
    >
      <h3 className="font-bold text-gray-900 mb-3">Accessibility</h3>

      <div className="space-y-3">
        {/* High Contrast Toggle */}
        <div className="flex items-center justify-between">
          <label
            htmlFor="high-contrast-toggle"
            className="text-sm flex items-center gap-2 text-gray-800"
          >
            {highContrast ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
            <span>High Contrast</span>
          </label>
          <button
            id="high-contrast-toggle"
            onClick={toggleHighContrast}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
              highContrast ? 'bg-blue-600' : 'bg-gray-300'
            }`}
            role="switch"
            aria-checked={highContrast}
            aria-label="Toggle high contrast mode"
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                highContrast ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </button>
        </div>

        {/* Font Size Controls */}
        <div>
          <label className="text-sm flex items-center gap-2 mb-2 text-gray-800">
            <Type className="w-4 h-4" />
            <span>Text Size</span>
          </label>
          <div
            className="flex gap-2"
            role="radiogroup"
            aria-label="Text size selection"
          >
            <button
              onClick={() => changeFontSize('normal')}
              className={`flex-1 px-3 py-2 rounded text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                fontSize === 'normal'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
              }`}
              role="radio"
              aria-checked={fontSize === 'normal'}
              aria-label="Normal text size"
            >
              A
            </button>
            <button
              onClick={() => changeFontSize('large')}
              className={`flex-1 px-3 py-2 rounded text-base font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                fontSize === 'large'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
              }`}
              role="radio"
              aria-checked={fontSize === 'large'}
              aria-label="Large text size"
            >
              A
            </button>
            <button
              onClick={() => changeFontSize('xlarge')}
              className={`flex-1 px-3 py-2 rounded text-lg font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                fontSize === 'xlarge'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
              }`}
              role="radio"
              aria-checked={fontSize === 'xlarge'}
              aria-label="Extra large text size"
            >
              A
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
