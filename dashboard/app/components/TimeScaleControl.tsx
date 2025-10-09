'use client';

import { useState } from 'react';
import { Clock, FastForward } from 'lucide-react';

type TimeScale = 1 | 2 | 3 | 4 | 5;

export default function TimeScaleControl() {
  const [activeScale, setActiveScale] = useState<TimeScale>(1);
  const [isChanging, setIsChanging] = useState(false);

  const handleScaleChange = async (scale: TimeScale) => {
    setIsChanging(true);
    try {
      const response = await fetch('/api/stream', {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ timeScale: scale }),
      });

      if (response.ok) {
        setActiveScale(scale);
      }
    } catch (error) {
      console.error('Failed to change time scale:', error);
    } finally {
      setTimeout(() => setIsChanging(false), 200);
    }
  };

  const scales: TimeScale[] = [1, 2, 3, 4, 5];

  return (
    <div className="bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-700 mt-6">
      <div className="mb-4">
        <div className="flex items-center gap-2 mb-2">
          <FastForward className="w-5 h-5 text-cyan-400" />
          <h3 className="text-lg font-bold text-white">Time Scale</h3>
        </div>
        <p className="text-gray-400 text-xs">
          Speed up simulation
        </p>
      </div>

      <div className="grid grid-cols-5 gap-2">
        {scales.map((scale) => (
          <button
            key={scale}
            onClick={() => handleScaleChange(scale)}
            disabled={isChanging}
            className={`
              relative p-3 rounded-lg transition-all duration-200
              ${activeScale === scale
                ? 'bg-gradient-to-br from-cyan-500 to-cyan-600 shadow-lg shadow-cyan-500/30 scale-105'
                : 'bg-gray-700 hover:bg-gray-650'
              }
              ${isChanging ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            <div className="text-center">
              <div className="text-white font-bold text-lg">
                {scale}x
              </div>
              {activeScale === scale && (
                <div className="absolute -top-1 -right-1">
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                </div>
              )}
            </div>
          </button>
        ))}
      </div>

      <div className="mt-4 flex items-center justify-center gap-2 text-xs">
        <Clock className="w-3 h-3 text-gray-500" />
        <span className="text-gray-500">
          {activeScale === 1 ? 'Real-time' : `${activeScale}x faster`}
        </span>
      </div>
    </div>
  );
}

