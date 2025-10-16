'use client';

import { useState } from 'react';
import { Clock, FastForward } from 'lucide-react';

type TimeScale = 1 | 2 | 3 | 4 | 5;

export default function TimeScaleControl() {
  const [activeScale, setActiveScale] = useState<TimeScale>(1);
  const [isChanging, setIsChanging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleScaleChange = async (scale: TimeScale) => {
    const previous = activeScale;
    setActiveScale(scale);
    setError(null);
    setIsChanging(true);
    try {
      const response = await fetch('/api/stream', {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ timeScale: scale }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        const message =
          typeof payload?.error === 'string'
            ? payload.error
            : `Failed to set time scale to ${scale}x`;
        throw new Error(message);
      }
    } catch (error) {
      console.error('Failed to change time scale:', error);
      setActiveScale(previous);
      setError('Unable to update time scale. Please retry.');
    } finally {
      setIsChanging(false);
    }
  };

  const scales: TimeScale[] = [1, 2, 3, 4, 5];

  return (
    <div className="bg-white rounded-xl p-6 shadow-md border border-slate-200 mt-6">
      <div className="mb-4">
        <div className="flex items-center gap-2 mb-2">
          <FastForward className="w-5 h-5 text-cyan-600" />
          <h3 className="text-lg font-bold text-slate-900">Time Scale</h3>
        </div>
        <p className="text-slate-500 text-xs">
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
                ? 'bg-gradient-to-br from-cyan-500 to-cyan-600 text-white shadow-lg shadow-cyan-500/30 scale-105 border border-cyan-500/70'
                : 'bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200'
              }
              ${isChanging ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            <div className="text-center">
              <div className={`font-bold text-lg ${activeScale === scale ? 'text-white' : 'text-slate-700'}`}>
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
        <Clock className="w-3 h-3 text-slate-400" />
        <span className="text-slate-500">
          {activeScale === 1 ? 'Real-time' : `${activeScale}x faster`}
        </span>
      </div>
      {error && (
        <div className="mt-3 text-xs text-rose-600 text-center bg-rose-50 border border-rose-200 rounded-md px-3 py-2">
          {error}
        </div>
      )}
    </div>
  );
}
