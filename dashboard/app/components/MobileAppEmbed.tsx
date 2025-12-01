"use client";

import { useState, useEffect } from "react";
import { Smartphone, X, ExternalLink } from "lucide-react";

interface MobileAppEmbedProps {
  isVisible: boolean;
  onClose: () => void;
  stressDetected: boolean;
}

export default function MobileAppEmbed({
  isVisible,
  onClose,
  stressDetected,
}: MobileAppEmbedProps) {
  const [mobileAppUrl, setMobileAppUrl] = useState<string>("");
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Start the mobile app on port 5173 (Vite default)
    setMobileAppUrl("http://localhost:5173");
    setIsLoading(false);
  }, []);

  const handleOpenInNewTab = () => {
    window.open(mobileAppUrl, "_blank");
  };

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-3xl shadow-2xl max-w-4xl w-full h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 p-4 text-white">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Smartphone className="w-6 h-6" />
              <div>
                <h3 className="font-bold text-lg">Mindwell Mobile App</h3>
                <p className="text-sm opacity-90">
                  {stressDetected
                    ? "🚨 Stress detected - Opening intervention app"
                    : "Mobile companion app"}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleOpenInNewTab}
                className="p-2 hover:bg-white/20 rounded-full transition-colors"
                title="Open in new tab"
              >
                <ExternalLink className="w-5 h-5" />
              </button>
              <button
                onClick={onClose}
                className="p-2 hover:bg-white/20 rounded-full transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Mobile App Container */}
        <div className="h-full relative">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-gray-600">Starting mobile app...</p>
              </div>
            </div>
          ) : (
            <iframe
              src={mobileAppUrl}
              className="w-full h-full border-0"
              title="Mindwell Mobile App"
              sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
            />
          )}
        </div>

        {/* Stress Detection Banner */}
        {stressDetected && (
          <div className="absolute top-20 left-4 right-4 bg-red-500 text-white p-3 rounded-lg shadow-lg animate-pulse">
            <div className="flex items-center gap-2">
              <span className="text-lg">🚨</span>
              <div>
                <p className="font-semibold">Stress Detected!</p>
                <p className="text-sm opacity-90">
                  The mobile app will guide you through stress relief exercises
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
