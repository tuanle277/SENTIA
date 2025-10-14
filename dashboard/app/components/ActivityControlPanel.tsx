'use client';

import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import {
  Smile,
  AlertTriangle,
} from 'lucide-react';

interface FeatureSnapshot {
  hrvMeanNN: number;
  edaMean: number;
  accMagMean: number;
  tempMean: number;
}

type StressMode = 'not_stressed' | 'stressed';

interface StressButton {
  mode: StressMode;
  label: string;
  icon: ReactNode;
  description: string;
  color: string;
}

interface ActivityControlPanelProps {
  latestFeatures: FeatureSnapshot | null;
}

const stressStates: StressButton[] = [
  {
    mode: 'not_stressed',
    label: 'Not Stressed',
    icon: <Smile className="w-5 h-5 text-white" />,
    description: 'Baseline physiological state',
    color: 'from-emerald-500 to-emerald-600'
  },
  {
    mode: 'stressed',
    label: 'Stressed',
    icon: <AlertTriangle className="w-5 h-5 text-white" />,
    description: 'Elevated stress response',
    color: 'from-rose-500 to-rose-600'
  }
];

const DEFAULT_PREDICTION_ENDPOINT = 'http://10.232.253.188:5000/predict';

export default function ActivityControlPanel({ latestFeatures }: ActivityControlPanelProps) {
  const [activeMode, setActiveMode] = useState<StressMode>('not_stressed');
  const [isChanging, setIsChanging] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState<string | null>(null);
  const [predictionDetails, setPredictionDetails] = useState<Record<string, unknown> | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const activeLabel = stressStates.find((state) => state.mode === activeMode)?.label ?? activeMode;

  const handleModeChange = async (mode: StressMode) => {
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
      console.error('Failed to change stress mode:', error);
    } finally {
      setTimeout(() => setIsChanging(false), 300);
    }
  };

  const featurePayload = useMemo(() => {
    if (!latestFeatures) return null;
    return {
      HRV_MeanNN: Number(latestFeatures.hrvMeanNN.toFixed(3)),
      EDA_Mean: Number(latestFeatures.edaMean.toFixed(3)),
      ACC_Mag_Mean: Number(latestFeatures.accMagMean.toFixed(3)),
      TEMP_Mean: Number(latestFeatures.tempMean.toFixed(3)),
    };
  }, [latestFeatures]);

  const runInference = useCallback(async () => {
    if (!featurePayload) {
      setPredictionError('Waiting for live telemetry before running inference.');
      return;
    }
    setIsPredicting(true);
    setPredictionError(null);
    setPredictionResult(null);
    setPredictionDetails(null);

    try {
      console.log('Sending prediction request with body:', JSON.stringify(featurePayload));
      const response = await fetch(DEFAULT_PREDICTION_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(featurePayload),
        cache: 'no-store',
      });

      if (!response.ok) {
        throw new Error(`Prediction request failed (${response.status})`);
      }

      const data = await response.json();
      const rawPrediction = (data?.prediction ?? data?.result ?? data) as unknown;

      let interpretedDetails: Record<string, unknown> | null = null;
      if (typeof rawPrediction === 'number') {
        setPredictionResult(rawPrediction === 1 ? 'Stressed' : 'Not stressed');
      } else if (typeof rawPrediction === 'string') {
        setPredictionResult(rawPrediction);
      } else if (rawPrediction && typeof rawPrediction === 'object') {
        const nested = rawPrediction as Record<string, unknown>;
        if (typeof nested.prediction === 'number') {
          setPredictionResult(nested.prediction === 1 ? 'Stressed' : 'Not stressed');
        } else if (typeof nested.label === 'string') {
          setPredictionResult(nested.label);
        } else {
          setPredictionResult('See details below');
        }
        interpretedDetails = nested;
      } else {
        setPredictionResult('Prediction received');
      }

      if (!interpretedDetails && data && typeof data === 'object') {
        interpretedDetails = data as Record<string, unknown>;
      }

      if (interpretedDetails) {
        setPredictionDetails(interpretedDetails);
      }
    } catch (error) {
      console.error('Failed to perform stress inference', error);
      setPredictionError('Failed to reach the stress inference service. Ensure FastAPI is running.');
    } finally {
      setIsPredicting(false);
    }
  }, [featurePayload]);

  useEffect(() => {
    if (!featurePayload) {
      return;
    }

    const interval = setInterval(() => {
      if (!isPredicting) {
        void runInference();
      }
    }, 500);

    return () => clearInterval(interval);
  }, [featurePayload, isPredicting, runInference]);

  return (
    <div className="bg-gray-800 rounded-2xl p-7 shadow-lg border border-gray-700 sticky top-8">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-white mb-3">Stress Simulator</h3>
        <p className="text-gray-400 text-sm leading-relaxed">
          Toggle between neutral and stressed physiology
        </p>
      </div>

      <div className="space-y-3">
        {stressStates.map((state) => (
          <button
            key={state.mode}
            onClick={() => handleModeChange(state.mode)}
            disabled={isChanging}
            className={`
              w-full p-4 rounded-lg transition-all duration-300
              ${activeMode === state.mode
                ? `bg-gradient-to-r ${state.color} shadow-lg scale-105`
                : 'bg-gray-700 hover:bg-gray-650 hover:scale-102'
              }
              ${isChanging ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
              disabled:cursor-not-allowed
            `}
          >
            <div className="flex items-center gap-3">
              <div className={`
                p-2 rounded-lg
                ${activeMode === state.mode
                  ? 'bg-white/20'
                  : 'bg-gray-600'
                }
              `}>
                {state.icon}
              </div>
              <div className="flex-1 text-left">
                <div className="font-semibold text-white">
                  {state.label}
                </div>
                <div className={`text-xs ${
                  activeMode === state.mode
                    ? 'text-white/80'
                    : 'text-gray-400'
                }`}>
                  {state.description}
                </div>
              </div>
              {activeMode === state.mode && (
                <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
              )}
            </div>
          </button>
        ))}
      </div>

      <div className="mt-7 p-5 bg-gray-900 rounded-xl border border-dashed border-gray-600">
        <h4 className="text-sm font-semibold text-gray-300 mb-3">Model Inference Preview</h4>
        <div className="space-y-4">
          <div className="rounded-lg border border-gray-700/60 bg-gray-800/60 p-4 text-xs text-gray-300">
            <div className="font-semibold text-gray-200 mb-2">Latest feature snapshot</div>
            {featurePayload ? (
              <dl className="grid grid-cols-2 gap-2">
                {Object.entries(featurePayload).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <dt className="uppercase tracking-wide text-gray-500">{key}</dt>
                    <dd className="font-mono text-gray-200">{value}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <p className="text-gray-500 text-center">Awaiting live telemetry...</p>
            )}
          </div>
          <button
            onClick={runInference}
            disabled={isPredicting || !featurePayload}
            className="w-full py-3 rounded-lg bg-gradient-to-r from-indigo-500 to-purple-500 text-white font-semibold hover:from-indigo-400 hover:to-purple-400 transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isPredicting ? 'Requesting prediction...' : 'Run Stress Inference'}
          </button>
          {predictionResult && (
            <div className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 p-4">
              <p className="text-sm font-semibold text-emerald-300">
                Prediction: <span className="text-white">{predictionResult}</span>
              </p>
              {predictionDetails && (
                <pre className="mt-3 text-xs text-gray-300 bg-gray-900/60 rounded-lg p-3 overflow-x-auto">
                  {JSON.stringify(predictionDetails, null, 2)}
                </pre>
              )}
            </div>
          )}
          {predictionError && (
            <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-300">
              {predictionError}
            </div>
          )}
        </div>
      </div>

      <div className="mt-7 p-5 bg-gray-900 rounded-xl border border-gray-700">
        <div className="text-xs text-gray-400 text-center">
          Current Mode
        </div>
        <div className="text-center text-white font-bold mt-2 text-lg">
          {activeLabel}
        </div>
      </div>
    </div>
  );
}
