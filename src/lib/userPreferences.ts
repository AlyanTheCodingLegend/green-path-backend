/**
 * User Preferences and Adaptive Learning System
 * Stores user preferences in localStorage for personalized experience
 * Implements adaptive intelligence by learning from user behavior
 */

export interface UserPreferences {
  preferredRouteType: 'cool' | 'fast' | null
  frequentLocations: Array<{
    name: string
    lat: number
    lon: number
    usageCount: number
  }>
  routeHistory: Array<{
    startLat: number
    startLon: number
    endLat: number
    endLon: number
    selectedRoute: 'cool' | 'fast'
    timestamp: string
  }>
  accessibilityPreferences: {
    highContrast: boolean
    fontSize: 'normal' | 'large' | 'xlarge'
  }
  privacyMode: boolean // Anonymous mode - no tracking
  lastCity: string | null
}

const DEFAULT_PREFERENCES: UserPreferences = {
  preferredRouteType: null,
  frequentLocations: [],
  routeHistory: [],
  accessibilityPreferences: {
    highContrast: false,
    fontSize: 'normal',
  },
  privacyMode: false,
  lastCity: null,
}

const STORAGE_KEY = 'greenpath_user_preferences'
const MAX_HISTORY = 50
const MAX_FREQUENT_LOCATIONS = 10

/**
 * Load user preferences from localStorage
 */
export function loadUserPreferences(): UserPreferences {
  if (typeof window === 'undefined') return DEFAULT_PREFERENCES

  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (!stored) return DEFAULT_PREFERENCES

    const parsed = JSON.parse(stored)
    return { ...DEFAULT_PREFERENCES, ...parsed }
  } catch (error) {
    console.error('Failed to load user preferences:', error)
    return DEFAULT_PREFERENCES
  }
}

/**
 * Save user preferences to localStorage
 */
export function saveUserPreferences(preferences: UserPreferences): void {
  if (typeof window === 'undefined') return

  try {
    // Don't save anything if privacy mode is enabled
    if (preferences.privacyMode) {
      localStorage.removeItem(STORAGE_KEY)
      return
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences))
  } catch (error) {
    console.error('Failed to save user preferences:', error)
  }
}

/**
 * Record a route selection to learn user preferences
 */
export function recordRouteSelection(
  startLat: number,
  startLon: number,
  endLat: number,
  endLon: number,
  selectedRoute: 'cool' | 'fast'
): void {
  const prefs = loadUserPreferences()

  // Don't record if in privacy mode
  if (prefs.privacyMode) return

  // Add to history
  prefs.routeHistory.unshift({
    startLat,
    startLon,
    endLat,
    endLon,
    selectedRoute,
    timestamp: new Date().toISOString(),
  })

  // Limit history size
  if (prefs.routeHistory.length > MAX_HISTORY) {
    prefs.routeHistory = prefs.routeHistory.slice(0, MAX_HISTORY)
  }

  // Update preferred route type based on recent history
  const recentSelections = prefs.routeHistory.slice(0, 10)
  const coolCount = recentSelections.filter((r) => r.selectedRoute === 'cool').length
  const fastCount = recentSelections.filter((r) => r.selectedRoute === 'fast').length

  if (coolCount > fastCount * 1.5) {
    prefs.preferredRouteType = 'cool'
  } else if (fastCount > coolCount * 1.5) {
    prefs.preferredRouteType = 'fast'
  }

  saveUserPreferences(prefs)
}

/**
 * Add a location to frequent locations
 */
export function addFrequentLocation(name: string, lat: number, lon: number): void {
  const prefs = loadUserPreferences()

  if (prefs.privacyMode) return

  // Check if location already exists (within ~100m)
  const existing = prefs.frequentLocations.find(
    (loc) => Math.abs(loc.lat - lat) < 0.001 && Math.abs(loc.lon - lon) < 0.001
  )

  if (existing) {
    existing.usageCount++
  } else {
    prefs.frequentLocations.push({ name, lat, lon, usageCount: 1 })
  }

  // Sort by usage count and limit size
  prefs.frequentLocations.sort((a, b) => b.usageCount - a.usageCount)
  if (prefs.frequentLocations.length > MAX_FREQUENT_LOCATIONS) {
    prefs.frequentLocations = prefs.frequentLocations.slice(0, MAX_FREQUENT_LOCATIONS)
  }

  saveUserPreferences(prefs)
}

/**
 * Get route recommendation based on user history
 */
export function getRouteRecommendation(): 'cool' | 'fast' | null {
  const prefs = loadUserPreferences()
  return prefs.preferredRouteType
}

/**
 * Update accessibility preferences
 * Note: Accessibility settings are always saved regardless of privacy mode
 */
export function updateAccessibilityPreferences(
  highContrast?: boolean,
  fontSize?: 'normal' | 'large' | 'xlarge'
): void {
  const prefs = loadUserPreferences()

  if (highContrast !== undefined) {
    prefs.accessibilityPreferences.highContrast = highContrast
  }

  if (fontSize !== undefined) {
    prefs.accessibilityPreferences.fontSize = fontSize
  }

  // Always save accessibility preferences even in privacy mode
  if (typeof window !== 'undefined') {
    try {
      const currentStored = localStorage.getItem(STORAGE_KEY)
      const currentPrefs = currentStored ? JSON.parse(currentStored) : {}
      currentPrefs.accessibilityPreferences = prefs.accessibilityPreferences
      localStorage.setItem(STORAGE_KEY, JSON.stringify(currentPrefs))
    } catch (error) {
      console.error('Failed to save accessibility preferences:', error)
    }
  }
}

/**
 * Toggle privacy mode
 */
export function setPrivacyMode(enabled: boolean): void {
  const prefs = loadUserPreferences()
  prefs.privacyMode = enabled

  if (enabled) {
    // Clear all tracking data when enabling privacy mode
    prefs.routeHistory = []
    prefs.frequentLocations = []
    prefs.preferredRouteType = null
    localStorage.removeItem(STORAGE_KEY)
  } else {
    saveUserPreferences(prefs)
  }
}

/**
 * Clear all user data
 */
export function clearAllUserData(): void {
  if (typeof window === 'undefined') return
  localStorage.removeItem(STORAGE_KEY)
}

/**
 * Set last used city
 */
export function setLastCity(cityName: string): void {
  const prefs = loadUserPreferences()
  if (!prefs.privacyMode) {
    prefs.lastCity = cityName
    saveUserPreferences(prefs)
  }
}

/**
 * Get statistics about user behavior for insights
 */
export function getUserStatistics(): {
  totalRoutes: number
  coolRoutesCount: number
  fastRoutesCount: number
  coolRoutePercentage: number
  mostUsedLocation: { name: string; count: number } | null
} {
  const prefs = loadUserPreferences()

  const coolCount = prefs.routeHistory.filter((r) => r.selectedRoute === 'cool').length
  const fastCount = prefs.routeHistory.filter((r) => r.selectedRoute === 'fast').length
  const total = prefs.routeHistory.length

  const mostUsed = prefs.frequentLocations[0] || null

  return {
    totalRoutes: total,
    coolRoutesCount: coolCount,
    fastRoutesCount: fastCount,
    coolRoutePercentage: total > 0 ? (coolCount / total) * 100 : 0,
    mostUsedLocation: mostUsed
      ? { name: mostUsed.name, count: mostUsed.usageCount }
      : null,
  }
}
