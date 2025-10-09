'use client';

import { useState } from 'react';
import { 
  PersonStanding, 
  Footprints, 
  Zap, 
  Dumbbell, 
  Moon 
} from 'lucide-react';

type ActivityMode = 'idle' | 'jogging' | 'sprinting' | 'basketball' | 'sleeping';

interface ActivityButton {
  mode: ActivityMode;
  label: string;
  icon: React.ReactNode;
  description: string;
  color: string;
}

const activities: ActivityButton[] = [
  {
    mode: 'idle',
    label: 'Idle',
    icon: <PersonStanding className="w-5 h-5 text-white" />,
    description: 'Resting state',
    color: 'from-blue-500 to-blue-600'
  },
  {
    mode: 'jogging',
    label: 'Jogging',
    icon: <Footprints className="w-5 h-5 text-white" />,
    description: 'Moderate cardio',
    color: 'from-green-500 to-green-600'
  },
  {
    mode: 'sprinting',
    label: 'Sprinting',
    icon: <Zap className="w-5 h-5 text-white" />,
    description: 'High intensity',
    color: 'from-red-500 to-red-600'
  },

  {
    mode: 'sleeping',
    label: 'Sleeping',
    icon: <Moon className="w-5 h-5 text-white" />,
    description: 'Deep rest',
    color: 'from-indigo-500 to-indigo-600'
  }
];

export default function ActivityControlPanel() {
  const [activeMode, setActiveMode] = useState<ActivityMode>('idle');
  const [isChanging, setIsChanging] = useState(false);

  const handleModeChange = async (mode: ActivityMode) => {
    setIsChanging(true);
    try {
      const response = await fetch('/api/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ mode }),
      });

      if (response.ok) {
        setActiveMode(mode);
      }
    } catch (error) {
      console.error('Failed to change activity mode:', error);
    } finally {
      setTimeout(() => setIsChanging(false), 300);
    }
  };

  return (
    <div className="bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-700 sticky top-8">
      <div className="mb-6">
        <h3 className="text-xl font-bold text-white mb-2">Activity Simulator</h3>
        <p className="text-gray-400 text-sm">
          Simulate different activity levels
        </p>
      </div>

      <div className="space-y-3">
        {activities.map((activity) => (
          <button
            key={activity.mode}
            onClick={() => handleModeChange(activity.mode)}
            disabled={isChanging}
            className={`
              w-full p-4 rounded-lg transition-all duration-300
              ${activeMode === activity.mode
                ? `bg-gradient-to-r ${activity.color} shadow-lg scale-105`
                : 'bg-gray-700 hover:bg-gray-650 hover:scale-102'
              }
              ${isChanging ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
              disabled:cursor-not-allowed
            `}
          >
            <div className="flex items-center gap-3">
              <div className={`
                p-2 rounded-lg
                ${activeMode === activity.mode
                  ? 'bg-white/20'
                  : 'bg-gray-600'
                }
              `}>
                {activity.icon}
              </div>
              <div className="flex-1 text-left">
                <div className="font-semibold text-white">
                  {activity.label}
                </div>
                <div className={`text-xs ${
                  activeMode === activity.mode
                    ? 'text-white/80'
                    : 'text-gray-400'
                }`}>
                  {activity.description}
                </div>
              </div>
              {activeMode === activity.mode && (
                <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
              )}
            </div>
          </button>
        ))}
      </div>

      <div className="mt-6 p-4 bg-gray-900 rounded-lg border border-gray-700">
        <div className="text-xs text-gray-400 text-center">
          Current Mode
        </div>
        <div className="text-center text-white font-bold mt-1 capitalize">
          {activeMode}
        </div>
      </div>
    </div>
  );
}

