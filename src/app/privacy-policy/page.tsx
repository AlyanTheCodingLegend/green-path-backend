import Link from 'next/link'
import { ArrowLeft, Shield } from 'lucide-react'

export default function PrivacyPolicyPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-blue-600 hover:text-blue-800 mb-3 transition-colors focus:outline-none focus:underline"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to GreenPath
          </Link>
          <div className="flex items-center gap-3">
            <Shield className="w-8 h-8 text-blue-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Privacy Policy</h1>
              <p className="text-sm text-gray-600">Last updated: December 2025</p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8">
        <div className="bg-white rounded-lg shadow-sm border p-8 space-y-6">
          <section>
            <h2 className="text-xl font-bold text-gray-900 mb-3">
              1. Introduction
            </h2>
            <p className="text-gray-700 leading-relaxed">
              GreenPath (&quot;we&quot;, &quot;our&quot;, &quot;us&quot;) is committed to protecting your privacy. This
              Privacy Policy explains how we handle your information when you use our
              thermal comfort route planning application designed specifically for
              Pakistani cities experiencing extreme heat conditions.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-gray-900 mb-3">
              2. Information We Collect
            </h2>
            <div className="space-y-3 text-gray-700">
              <div>
                <h3 className="font-semibold mb-1">2.1 Locally Stored Data</h3>
                <p className="leading-relaxed">
                  All user preferences and usage data are stored locally in your browser
                  using localStorage. This includes:
                </p>
                <ul className="list-disc list-inside ml-4 mt-2 space-y-1">
                  <li>Route preferences (cool vs. fast route selections)</li>
                  <li>Frequently used locations (coordinates and names)</li>
                  <li>Route history (start/end points and selected route types)</li>
                  <li>Accessibility preferences (high contrast mode, font size)</li>
                  <li>Last selected city</li>
                </ul>
              </div>

              <div>
                <h3 className="font-semibold mb-1">2.2 Data NOT Collected</h3>
                <p className="leading-relaxed">
                  We do NOT collect, transmit, or store on any server:
                </p>
                <ul className="list-disc list-inside ml-4 mt-2 space-y-1">
                  <li>Personal identification information</li>
                  <li>Email addresses or contact information</li>
                  <li>Device identifiers</li>
                  <li>IP addresses</li>
                  <li>Location tracking outside of explicit route queries</li>
                </ul>
              </div>
            </div>
          </section>

          <section>
            <h2 className="text-xl font-bold text-gray-900 mb-3">
              3. How We Use Your Information
            </h2>
            <p className="text-gray-700 leading-relaxed mb-2">
              The locally stored information is used solely to:
            </p>
            <ul className="list-disc list-inside ml-4 space-y-1 text-gray-700">
              <li>Provide personalized route recommendations based on your preferences</li>
              <li>Remember your frequently used locations for quick access</li>
              <li>Learn and adapt to your route preferences over time</li>
              <li>Maintain your accessibility settings across sessions</li>
              <li>Display statistics about your sustainable route choices</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-bold text-gray-900 mb-3">
              4. Anonymous Mode
            </h2>
            <p className="text-gray-700 leading-relaxed">
              You can enable Anonymous Mode at any time in the Privacy Settings. When
              enabled:
            </p>
            <ul className="list-disc list-inside ml-4 mt-2 space-y-1 text-gray-700">
              <li>All existing locally stored data is immediately cleared</li>
              <li>No new usage data or preferences will be saved</li>
              <li>The app functions normally but without personalization</li>
              <li>You can disable Anonymous Mode to resume data collection</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-bold text-gray-900 mb-3">
              5. Data Sharing and Third Parties
            </h2>
            <p className="text-gray-700 leading-relaxed mb-2">
              <strong>We do not share your data with any third parties.</strong> However,
              please note:
            </p>
            <ul className="list-disc list-inside ml-4 space-y-1 text-gray-700">
              <li>
                Satellite imagery data is fetched from Google Earth Engine for thermal
                comfort calculations
              </li>
              <li>
                Map tiles are loaded from OpenStreetMap for route visualization
              </li>
              <li>These services receive only the coordinates you explicitly query</li>
              <li>No personal information is transmitted to these services</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-bold text-gray-900 mb-3">
              6. Data Security
            </h2>
            <p className="text-gray-700 leading-relaxed">
              Your data is stored locally in your browser's localStorage. This means:
            </p>
            <ul className="list-disc list-inside ml-4 mt-2 space-y-1 text-gray-700">
              <li>Data never leaves your device</li>
              <li>You have full control over your data</li>
              <li>You can clear all data at any time using the "Clear All Data" button</li>
              <li>Clearing your browser's cache/cookies will also clear this data</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-bold text-gray-900 mb-3">
              7. Your Rights and Control
            </h2>
            <p className="text-gray-700 leading-relaxed mb-2">You have the right to:</p>
            <ul className="list-disc list-inside ml-4 space-y-1 text-gray-700">
              <li>Access your locally stored data via browser developer tools</li>
              <li>Delete all your data using the "Clear All Data" button</li>
              <li>Use the app in Anonymous Mode with no data collection</li>
              <li>Export your data (browser localStorage can be accessed manually)</li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-bold text-gray-900 mb-3">
              8. Children's Privacy
            </h2>
            <p className="text-gray-700 leading-relaxed">
              GreenPath does not knowingly collect any information from children under 13.
              The app is designed for general public use in urban Pakistani cities,
              particularly for pedestrians seeking thermally comfortable walking routes
              during hot weather.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-gray-900 mb-3">
              9. Cookies and Tracking
            </h2>
            <p className="text-gray-700 leading-relaxed">
              GreenPath does not use cookies or any tracking technologies. All data
              storage is through browser localStorage which you can clear at any time.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-gray-900 mb-3">
              10. Changes to This Policy
            </h2>
            <p className="text-gray-700 leading-relaxed">
              We may update this Privacy Policy from time to time. Any changes will be
              posted on this page with an updated "Last updated" date. We encourage you
              to review this policy periodically.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-gray-900 mb-3">
              11. Ethical Considerations
            </h2>
            <p className="text-gray-700 leading-relaxed mb-2">
              GreenPath is designed with the following ethical principles:
            </p>
            <ul className="list-disc list-inside ml-4 space-y-1 text-gray-700">
              <li>
                <strong>Public Health Focus:</strong> Helping vulnerable populations avoid
                heat stress in Pakistani cities
              </li>
              <li>
                <strong>Environmental Awareness:</strong> Promoting tree-lined routes
                supports urban forest preservation
              </li>
              <li>
                <strong>Accessibility First:</strong> Designed with features for users
                with disabilities
              </li>
              <li>
                <strong>Transparency:</strong> All algorithms and data sources are
                documented
              </li>
              <li>
                <strong>No Discrimination:</strong> Routes are calculated objectively
                based on thermal comfort metrics
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-bold text-gray-900 mb-3">12. Contact</h2>
            <p className="text-gray-700 leading-relaxed">
              If you have any questions about this Privacy Policy or how your data is
              handled, please contact us through the GreenPath GitHub repository or reach
              out to your local municipal authorities regarding urban heat mitigation
              initiatives.
            </p>
          </section>

          <div className="mt-8 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-900">
              <strong>Summary:</strong> GreenPath stores your preferences locally on your
              device only. No data is sent to our servers. You can clear all data anytime
              or use Anonymous Mode for zero data collection. Your privacy is our
              priority.
            </p>
          </div>
        </div>
      </main>
    </div>
  )
}
