'use client'

import { useEffect, useState } from 'react'
import { ProgressEvent } from '@/lib/api'
import { Loader2, CheckCircle2, XCircle } from 'lucide-react'

interface ProgressDialogProps {
  isOpen: boolean
  events: ProgressEvent[]
  onClose: () => void
}

export default function ProgressDialog({ isOpen, events, onClose }: ProgressDialogProps) {
  const [isComplete, setIsComplete] = useState(false)
  const [hasError, setHasError] = useState(false)

  useEffect(() => {
    const lastEvent = events[events.length - 1]
    if (lastEvent) {
      setIsComplete(!!lastEvent.complete)
      setHasError(!!lastEvent.error)
    }
  }, [events])

  if (!isOpen) return null

  const latestEvent = events[events.length - 1]
  const progress = latestEvent?.progress || 0

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-bold">
            {isComplete ? 'Complete' : hasError ? 'Error' : 'Loading City Data'}
          </h2>
          {isComplete && <CheckCircle2 className="w-6 h-6 text-green-500" />}
          {hasError && <XCircle className="w-6 h-6 text-red-500" />}
        </div>

        {/* Progress bar */}
        {!isComplete && !hasError && (
          <div className="space-y-2">
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="text-sm text-gray-600 text-right">{progress}%</div>
          </div>
        )}

        {/* Progress messages */}
        <div className="max-h-48 overflow-y-auto space-y-2 bg-gray-50 rounded p-3">
          {events.map((event, index) => (
            <div key={index} className="flex items-start gap-2 text-sm">
              {!event.complete && !event.error && (
                <Loader2 className="w-4 h-4 animate-spin text-blue-500 mt-0.5 flex-shrink-0" />
              )}
              {event.complete && (
                <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
              )}
              {event.error && (
                <XCircle className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
              )}
              <span className={event.error ? 'text-red-600' : 'text-gray-700'}>
                {event.message}
              </span>
            </div>
          ))}
        </div>

        {/* Close button */}
        {(isComplete || hasError) && (
          <button
            onClick={onClose}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded transition-colors"
          >
            Close
          </button>
        )}
      </div>
    </div>
  )
}
