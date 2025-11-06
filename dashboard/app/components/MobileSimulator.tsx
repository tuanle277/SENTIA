"use client";

import { useState, useEffect } from "react";
import { Smartphone, X, Play, Pause, RotateCcw } from "lucide-react";

interface MobileSimulatorProps {
  isVisible: boolean;
  onClose: () => void;
  stressDetected: boolean;
}

export default function MobileSimulator({
  isVisible,
  onClose,
  stressDetected,
}: MobileSimulatorProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [timeRemaining, setTimeRemaining] = useState(0);

  const breathingSteps = [
    { instruction: "Find a comfortable position", duration: 10 },
    { instruction: "Breathe in slowly for 4 counts", duration: 4 },
    { instruction: "Hold your breath for 4 counts", duration: 4 },
    { instruction: "Breathe out slowly for 6 counts", duration: 6 },
    { instruction: "Repeat this cycle", duration: 0 },
  ];

  useEffect(() => {
    if (stressDetected && isVisible) {
      setIsPlaying(true);
      setCurrentStep(0);
      setTimeRemaining(breathingSteps[0].duration);
    }
  }, [stressDetected, isVisible]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying && timeRemaining > 0) {
      interval = setInterval(() => {
        setTimeRemaining((prev) => {
          if (prev <= 1) {
            if (currentStep < breathingSteps.length - 1) {
              setCurrentStep(currentStep + 1);
              return breathingSteps[currentStep + 1].duration;
            } else {
              setIsPlaying(false);
              return 0;
            }
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isPlaying, timeRemaining, currentStep]);

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentStep(0);
    setTimeRemaining(breathingSteps[0].duration);
  };

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-3xl shadow-2xl max-w-md w-full overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 p-4 text-white">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Smartphone className="w-6 h-6" />
              <div>
                <h3 className="font-bold text-lg">Mindwell Companion</h3>
                <p className="text-sm opacity-90">Mobile App Simulator</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-white/20 rounded-full transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Mobile Screen */}
        <div className="p-6">
          {stressDetected ? (
            <div className="text-center">
              <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">🚨</span>
              </div>
              <h2 className="text-xl font-bold text-gray-900 mb-2">
                Stress Detected
              </h2>
              <p className="text-gray-600 mb-6">
                Your physiological indicators suggest elevated stress levels.
                Let's help you reset.
              </p>

              {/* Breathing Exercise */}
              <div className="bg-gray-50 rounded-2xl p-6 mb-6">
                <h3 className="font-semibold text-gray-900 mb-4">
                  Guided Breathing Exercise
                </h3>
                <div className="text-center">
                  <p className="text-lg text-gray-700 mb-4">
                    {breathingSteps[currentStep].instruction}
                  </p>
                  {timeRemaining > 0 && (
                    <div className="text-4xl font-bold text-blue-600 mb-4">
                      {timeRemaining}
                    </div>
                  )}
                  <div className="flex justify-center gap-3">
                    <button
                      onClick={handlePlayPause}
                      className="px-4 py-2 bg-blue-600 text-white rounded-lg flex items-center gap-2 hover:bg-blue-700 transition-colors"
                    >
                      {isPlaying ? (
                        <Pause className="w-4 h-4" />
                      ) : (
                        <Play className="w-4 h-4" />
                      )}
                      {isPlaying ? "Pause" : "Start"}
                    </button>
                    <button
                      onClick={handleReset}
                      className="px-4 py-2 bg-gray-600 text-white rounded-lg flex items-center gap-2 hover:bg-gray-700 transition-colors"
                    >
                      <RotateCcw className="w-4 h-4" />
                      Reset
                    </button>
                  </div>
                </div>
              </div>

              <div className="text-sm text-gray-500">
                Step {currentStep + 1} of {breathingSteps.length}
              </div>
            </div>
          ) : (
            <div className="text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">😌</span>
              </div>
              <h2 className="text-xl font-bold text-gray-900 mb-2">All Good</h2>
              <p className="text-gray-600">
                Your physiological indicators are within normal ranges.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
