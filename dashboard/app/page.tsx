'use client';

import { useState, useEffect, useCallback } from 'react';
import MetricCard from './components/MetricCard';
import RealtimeChart from './components/RealtimeChart';
import ActivityIndicator from './components/ActivityIndicator';
import ActivityControlPanel from './components/ActivityControlPanel';
import TimeScaleControl from './components/TimeScaleControl';
import { 
  Heart, 
  Brain, 
  Droplet, 
  Thermometer, 
  Footprints, 
  Flame,
  Moon,
  Activity
} from 'lucide-react';

interface WearableData {
  timestamp: number;
  heartRate: number;
  stressLevel: number;
  spo2: number;
  temperature: number;
  steps: number;
  calories: number;
  sleepQuality: number;
  activityLevel: 'resting' | 'light' | 'moderate' | 'intense';
}

interface ChartDataPoint {
  timestamp: number;
  value: number;
}

const MAX_DATA_POINTS = 60; 

export default function Dashboard() {
  const [currentData, setCurrentData] = useState<WearableData | null>(null);
  const [heartRateHistory, setHeartRateHistory] = useState<ChartDataPoint[]>([]);
  const [stressHistory, setStressHistory] = useState<ChartDataPoint[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');

  const connectToStream = useCallback(() => {
    const eventSource = new EventSource('/api/stream');

    eventSource.onopen = () => {
      setConnectionStatus('connected');
    };

    eventSource.onmessage = (event) => {
      try {
        const data: WearableData = JSON.parse(event.data);
        setCurrentData(data);

        
        setHeartRateHistory((prev) => {
          const newData = [...prev, { timestamp: data.timestamp, value: data.heartRate }];
          return newData.slice(-MAX_DATA_POINTS);
        });

        setStressHistory((prev) => {
          const newData = [...prev, { timestamp: data.timestamp, value: data.stressLevel }];
          return newData.slice(-MAX_DATA_POINTS);
        });
      } catch (error) {
        console.error('Error parsing stream data:', error);
      }
    };

    eventSource.onerror = () => {
      setConnectionStatus('disconnected');
      eventSource.close();
      
      
      setTimeout(() => {
        setConnectionStatus('connecting');
        connectToStream();
      }, 3000);
    };

    return eventSource;
  }, []);

  useEffect(() => {
    const eventSource = connectToStream();
    return () => eventSource.close();
  }, [connectToStream]);

  const getHeartRateStatus = (hr: number) => {
    if (hr < 60 || hr > 100) return 'warning';
    if (hr > 120) return 'danger';
    return 'normal';
  };

  const getStressStatus = (stress: number) => {
    if (stress > 70) return 'danger';
    if (stress > 50) return 'warning';
    return 'normal';
  };

  const getSpo2Status = (spo2: number) => {
    if (spo2 < 95) return 'warning';
    if (spo2 < 90) return 'danger';
    return 'normal';
  };

  const getTempStatus = (temp: number) => {
    if (temp > 37.5 || temp < 36) return 'warning';
    if (temp > 38) return 'danger';
    return 'normal';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex justify-center">
      <div className="w-full max-w-[1920px] px-8 py-8">
        {/* Header */}
        <div className="mb-10">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">
                Health Dashboard
              </h1>
              <p className="text-gray-400">Real-time wearable metrics monitoring</p>
            </div>
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${
                connectionStatus === 'connected' ? 'bg-green-500 animate-pulse' : 
                connectionStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' : 
                'bg-red-500'
              }`}></div>
              <span className="text-gray-400 text-sm capitalize">{connectionStatus}</span>
            </div>
          </div>
        </div>

        {/* Main Layout: Dashboard + Control Panel */}
        <div className="flex gap-8">
          {/* Main Dashboard Content */}
          <div className="flex-1 min-w-0">
            {currentData ? (
              <>
            {/* Primary Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-8">
              <MetricCard
                title="Heart Rate"
                value={currentData.heartRate}
                unit="BPM"
                icon={<Heart className="w-6 h-6 text-red-400" />}
                status={getHeartRateStatus(currentData.heartRate)}
              />
              <MetricCard
                title="Stress Level"
                value={currentData.stressLevel}
                unit="%"
                icon={<Brain className="w-6 h-6 text-purple-400" />}
                status={getStressStatus(currentData.stressLevel)}
              />
              <MetricCard
                title="Blood Oxygen"
                value={currentData.spo2}
                unit="%"
                icon={<Droplet className="w-6 h-6 text-blue-400" />}
                status={getSpo2Status(currentData.spo2)}
                subtitle="SpO2"
              />
              <MetricCard
                title="Temperature"
                value={currentData.temperature}
                unit="°C"
                icon={<Thermometer className="w-6 h-6 text-orange-400" />}
                status={getTempStatus(currentData.temperature)}
              />
            </div>

            {/* Secondary Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-8">
              <MetricCard
                title="Steps"
                value={currentData.steps.toLocaleString()}
                icon={<Footprints className="w-6 h-6 text-green-400" />}
                subtitle="Today"
              />
              <MetricCard
                title="Calories"
                value={currentData.calories.toLocaleString()}
                unit="kcal"
                icon={<Flame className="w-6 h-6 text-orange-400" />}
                subtitle="Burned"
              />
              <MetricCard
                title="Sleep Quality"
                value={currentData.sleepQuality}
                unit="%"
                icon={<Moon className="w-6 h-6 text-indigo-400" />}
                subtitle="Last night"
              />
              <div className="md:hidden lg:block">
                <ActivityIndicator level={currentData.activityLevel} />
              </div>
            </div>

            {/* Activity Indicator for medium screens */}
            <div className="hidden md:block lg:hidden mb-8">
              <ActivityIndicator level={currentData.activityLevel} />
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <RealtimeChart
                data={heartRateHistory}
                title="Heart Rate Trend"
                color="#EF4444"
                unit=" BPM"
                domain={[50, 180]}
              />
              <RealtimeChart
                data={stressHistory}
                title="Stress Level Trend"
                color="#A855F7"
                unit="%"
                domain={[0, 100]}
              />
            </div>

      
              </>
            ) : (
              <div className="flex items-center justify-center h-96">
                <div className="text-center">
                  <Activity className="w-16 h-16 text-gray-600 animate-pulse mx-auto mb-4" />
                  <p className="text-gray-400 text-lg">Connecting to wearable device...</p>
                </div>
              </div>
            )}
          </div>

          {/* Activity Control Panel */}
          <div className="w-80 flex-shrink-0">
            <ActivityControlPanel />
            <TimeScaleControl />
          </div>
        </div>
      </div>
    </div>
  );
}
