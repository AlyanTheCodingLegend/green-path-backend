import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDistance(km: number): string {
  return `${km.toFixed(2)} km`
}

export function formatTime(minutes: number): string {
  const hours = Math.floor(minutes / 60)
  const mins = Math.round(minutes % 60)

  if (hours > 0) {
    return `${hours}h ${mins}m`
  }
  return `${mins}m`
}

export function formatComfort(score: number): string {
  return `${(score * 100).toFixed(0)}%`
}

export function getComfortColor(score: number): string {
  if (score < 0.3) return '#ef4444' // red
  if (score < 0.5) return '#f97316' // orange
  if (score < 0.7) return '#eab308' // yellow
  if (score < 0.85) return '#84cc16' // lime
  return '#22c55e' // green
}

export function getComfortCategory(score: number): string {
  if (score < 0.3) return 'Poor'
  if (score < 0.5) return 'Fair'
  if (score < 0.7) return 'Good'
  return 'Excellent'
}
