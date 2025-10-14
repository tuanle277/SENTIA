'use client';

import { useState, useEffect, useCallback } from 'react';
import MetricCard from './components/MetricCard';
import RealtimeChart from './components/RealtimeChart';
import ActivityControlPanel from './components/ActivityControlPanel';
import TimeScaleControl from './components/TimeScaleControl';
import {
  Activity,
  Droplet,
  Gauge,
  Thermometer,
  Loader2,
} from 'lucide-react';

interface FeatureStreamData {
  timestamp: number;
  hrvMeanNN: number;
  edaMean: number;
  accMagMean: number;
  tempMean: number;
  stressLabel?: number;
}

interface ChartDataPoint {
  timestamp: number;
  value: number;
}

const MAX_DATA_POINTS = 60; 

const FEATURE_BASELINES = {
  hrvMeanNN: { mean: 821.286, std: 55.89 },
  edaMean: { mean: 1.13, std: 0.22 },
  accMagMean: { mean: 63.51, std: 0.488 },
  tempMean: { mean: 35.762, std: 0.053 },
};

export default function Dashboard() {
  const [currentData, setCurrentData] = useState<FeatureStreamData | null>(null);
  const [hrvHistory, setHrvHistory] = useState<ChartDataPoint[]>([]);
  const [edaHistory, setEdaHistory] = useState<ChartDataPoint[]>([]);
  const [accHistory, setAccHistory] = useState<ChartDataPoint[]>([]);
  const [tempHistory, setTempHistory] = useState<ChartDataPoint[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');

  const connectToStream = useCallback(() => {
    const eventSource = new EventSource('/api/stream');

    eventSource.onopen = () => {
      setConnectionStatus('connected');
    };

    eventSource.onmessage = (event) => {
      try {
        const data: FeatureStreamData = JSON.parse(event.data);
        setCurrentData(data);

        setHrvHistory((prev) => {
          const newData = [...prev, { timestamp: data.timestamp, value: data.hrvMeanNN }];
          return newData.slice(-MAX_DATA_POINTS);
        });

        setEdaHistory((prev) => {
          const newData = [...prev, { timestamp: data.timestamp, value: data.edaMean }];
          return newData.slice(-MAX_DATA_POINTS);
        });

        setAccHistory((prev) => {
          const newData = [...prev, { timestamp: data.timestamp, value: data.accMagMean }];
          return newData.slice(-MAX_DATA_POINTS);
        });

        setTempHistory((prev) => {
          const newData = [...prev, { timestamp: data.timestamp, value: data.tempMean }];
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

  const getHrvStatus = (value: number) => {
    const baseline = FEATURE_BASELINES.hrvMeanNN;
    if (value < baseline.mean - baseline.std * 1.3) return 'danger';
    if (value < baseline.mean - baseline.std * 0.7) return 'warning';
    return 'normal';
  };

  const getEdaStatus = (value: number) => {
    const baseline = FEATURE_BASELINES.edaMean;
    if (value > baseline.mean + baseline.std * 1.3) return 'danger';
    if (value > baseline.mean + baseline.std * 0.7) return 'warning';
    return 'normal';
  };

  const getAccStatus = (value: number) => {
    const baseline = FEATURE_BASELINES.accMagMean;
    if (value > baseline.mean + baseline.std * 1.6) return 'danger';
    if (value > baseline.mean + baseline.std * 0.9) return 'warning';
    return 'normal';
  };

  const getTempStatus = (value: number) => {
    const baseline = FEATURE_BASELINES.tempMean;
    const diff = value - baseline.mean;
    if (Math.abs(diff) > baseline.std * 3) return 'danger';
    if (Math.abs(diff) > baseline.std * 2) return 'warning';
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
                title="HRV Mean NN"
                value={currentData.hrvMeanNN.toFixed(0)}
                unit="ms"
                icon={<Activity className="w-6 h-6 text-cyan-400" />}
                status={getHrvStatus(currentData.hrvMeanNN)}
              />
              <MetricCard
                title="EDA Mean"
                value={currentData.edaMean.toFixed(3)}
                unit="uS"
                icon={<Droplet className="w-6 h-6 text-blue-400" />}
                status={getEdaStatus(currentData.edaMean)}
              />
              <MetricCard
                title="ACC Magnitude"
                value={currentData.accMagMean.toFixed(2)}
                unit="g"
                icon={<Gauge className="w-6 h-6 text-purple-400" />}
                status={getAccStatus(currentData.accMagMean)}
              />
              <MetricCard
                title="Skin Temperature"
                value={currentData.tempMean.toFixed(2)}
                unit="°C"
                icon={<Thermometer className="w-6 h-6 text-orange-400" />}
                status={getTempStatus(currentData.tempMean)}
              />
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <RealtimeChart
                data={hrvHistory}
                title="HRV Mean NN Trend"
                color="#22D3EE"
                unit=" ms"
                domain={[
                  FEATURE_BASELINES.hrvMeanNN.mean - FEATURE_BASELINES.hrvMeanNN.std * 2,
                  FEATURE_BASELINES.hrvMeanNN.mean + FEATURE_BASELINES.hrvMeanNN.std * 2,
                ]}
              />
              <RealtimeChart
                data={edaHistory}
                title="EDA Mean Trend"
                color="#F97316"
                unit=" uS"
                domain={[
                  Math.max(0, FEATURE_BASELINES.edaMean.mean - FEATURE_BASELINES.edaMean.std),
                  FEATURE_BASELINES.edaMean.mean + FEATURE_BASELINES.edaMean.std * 3,
                ]}
              />
              <RealtimeChart
                data={accHistory}
                title="ACC Magnitude Trend"
                color="#A855F7"
                unit=" g"
                domain={[
                  FEATURE_BASELINES.accMagMean.mean - FEATURE_BASELINES.accMagMean.std * 2,
                  FEATURE_BASELINES.accMagMean.mean + FEATURE_BASELINES.accMagMean.std * 2,
                ]}
              />
              <RealtimeChart
                data={tempHistory}
                title="Skin Temperature Trend"
                color="#FACC15"
                unit=" °C"
                domain={[
                  FEATURE_BASELINES.tempMean.mean - FEATURE_BASELINES.tempMean.std * 3,
                  FEATURE_BASELINES.tempMean.mean + FEATURE_BASELINES.tempMean.std * 3,
                ]}
              />
            </div>

      
              </>
            ) : (
              <div className="flex items-center justify-center h-96">
                <div className="text-center">
                  <Loader2 className="w-16 h-16 text-gray-600 animate-spin mx-auto mb-4" />
                  <p className="text-gray-400 text-lg">Connecting to wearable device...</p>
                </div>
              </div>
            )}
          </div>

          {/* Activity Control Panel */}
          <div className="w-[420px] flex-shrink-0">
            <ActivityControlPanel latestFeatures={currentData} />
            <TimeScaleControl />
          </div>
        </div>
      </div>
    </div>
  );
}
