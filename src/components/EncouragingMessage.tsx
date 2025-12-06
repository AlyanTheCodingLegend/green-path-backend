'use client'

import { useEffect, useState } from 'react'
import { Sparkles, Heart, Leaf, ThumbsUp, LucideIcon } from 'lucide-react'

interface EncouragingMessageProps {
  comfortImprovement: number // Percentage improvement
  distancePenalty: number // Percentage increase in distance
  coolRouteSelected: boolean
}

interface MessageItem {
  text: string
  icon: LucideIcon
}

const MESSAGES: Record<string, MessageItem[]> = {
  highComfort: [
    { text: 'Excellent choice! This route will keep you much cooler.', icon: Sparkles },
    { text: 'Great decision! Your body will thank you on this shaded path.', icon: Heart },
    { text: 'Smart pick! More trees mean a more comfortable journey.', icon: Leaf },
  ],
  mediumComfort: [
    { text: 'Good choice! A bit more distance for better comfort.', icon: ThumbsUp },
    { text: 'Nice! Trading a few extra meters for shade is worth it.', icon: Leaf },
  ],
  lowDistance: [
    { text: 'Perfect balance of comfort and efficiency!', icon: Sparkles },
    { text: 'Best of both worlds - cool and quick!', icon: ThumbsUp },
  ],
  environmental: [
    { text: 'Choosing green routes helps preserve urban forests!', icon: Leaf },
    { text: 'Every shaded route choice supports sustainable cities.', icon: Heart },
  ],
}

export default function EncouragingMessage({
  comfortImprovement,
  distancePenalty,
  coolRouteSelected,
}: EncouragingMessageProps) {
  const [message, setMessage] = useState<MessageItem | null>(null)
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    if (!coolRouteSelected) {
      setIsVisible(false)
      return
    }

    // Select appropriate encouraging message based on metrics
    let selectedMessage

    if (comfortImprovement > 15 && distancePenalty < 20) {
      // Excellent choice: high comfort gain, low distance penalty
      selectedMessage =
        MESSAGES.lowDistance[Math.floor(Math.random() * MESSAGES.lowDistance.length)]
    } else if (comfortImprovement > 10) {
      // High comfort improvement
      selectedMessage =
        MESSAGES.highComfort[Math.floor(Math.random() * MESSAGES.highComfort.length)]
    } else if (comfortImprovement > 5) {
      // Medium comfort improvement
      selectedMessage =
        MESSAGES.mediumComfort[
          Math.floor(Math.random() * MESSAGES.mediumComfort.length)
        ]
    } else {
      // Any cool route choice helps environment
      selectedMessage =
        MESSAGES.environmental[
          Math.floor(Math.random() * MESSAGES.environmental.length)
        ]
    }

    setMessage(selectedMessage)
    setIsVisible(true)

    // Auto-hide after 5 seconds
    const timer = setTimeout(() => {
      setIsVisible(false)
    }, 5000)

    return () => clearTimeout(timer)
  }, [comfortImprovement, distancePenalty, coolRouteSelected])

  if (!message || !isVisible) return null

  const Icon = message.icon

  return (
    <div
      className="bg-green-100 border-2 border-green-400 rounded-lg p-4 mb-4 animate-fade-in"
      role="status"
      aria-live="polite"
    >
      <div className="flex items-start gap-3">
        <div className="w-10 h-10 bg-green-600 rounded-full flex items-center justify-center flex-shrink-0">
          <Icon className="w-5 h-5 text-white" aria-hidden="true" />
        </div>
        <div className="flex-1">
          <p className="text-green-900 font-medium">{message.text}</p>
          <div className="mt-2 flex gap-4 text-sm text-green-800">
            <span>
              <strong>+{comfortImprovement.toFixed(1)}%</strong> comfort
            </span>
            {distancePenalty > 0 && (
              <span>
                <strong>+{distancePenalty.toFixed(1)}%</strong> distance
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
