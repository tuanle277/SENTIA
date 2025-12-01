"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { Smile, AlertTriangle } from "lucide-react";

interface FeatureSnapshot {
  hrvMeanNN: number;
  edaMean: number;
  accMagMean: number;
  tempMean: number;
  stressLabel?: number;
}

type StressMode = "not_stressed" | "stressed";
type StreamSource =
  | "simulated"
  | "real_all"
  | "real_not_stressed"
  | "real_stressed";

interface StreamMeta {
  source: StreamSource;
  availableRealSamples?: {
    all: number;
    notStressed: number;
    stressed: number;
  };
}

interface StressButton {
  mode: StressMode;
  label: string;
  icon: ReactNode;
  description: string;
  color: string;
}

interface ActivityControlPanelProps {
  latestFeatures: FeatureSnapshot | null;
  streamMeta: StreamMeta | null;
  onStressTrigger?: (timestamp: number) => void;
  monitoringPaused?: boolean;
}

const stressStates: StressButton[] = [
  {
    mode: "stressed",
    label: "Not Stressed",
    icon: <Smile className="w-5 h-5 text-white" />,
    description: "Baseline physiological state",
    color: "from-emerald-500 to-emerald-600",
  },
  {
    mode: "not_stressed",
    label: "Stressed",
    icon: <AlertTriangle className="w-5 h-5 text-white" />,
    description: "Elevated stress response",
    color: "from-rose-500 to-rose-600",
  },
];

const DEFAULT_PREDICTION_ENDPOINT =
  process.env.NEXT_PUBLIC_STRESS_API_URL ?? "http://127.0.0.1:5000/predict";

const REAL_SOURCE_LABELS: Record<Exclude<StreamSource, "simulated">, string> = {
  real_all: "Real Stream (All Samples)",
  real_not_stressed: "Real Stream – Not Stressed",
  real_stressed: "Real Stream – Stressed",
};

const describeRealSource = (source: StreamSource): string | null => {
  if (source === "simulated") return null;
  return REAL_SOURCE_LABELS[source];
};

export default function ActivityControlPanel({
  latestFeatures,
  streamMeta,
  onStressTrigger,
  monitoringPaused = false,
}: ActivityControlPanelProps) {
  const [activeMode, setActiveMode] = useState<StressMode>("not_stressed");
  const [isChanging, setIsChanging] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState<string | null>(null);
  const [predictionDetails, setPredictionDetails] = useState<Record<
    string,
    unknown
  > | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const lastTriggerRef = useRef<number>(0);
  const lastWasStressedRef = useRef<boolean>(false);
  const [streamSource, setStreamSource] = useState<StreamSource>(
    streamMeta?.source ?? "simulated"
  );
  const activeLabel =
    stressStates.find((state) => state.mode === activeMode)?.label ??
    activeMode;

  useEffect(() => {
    if (streamMeta?.source) {
      setStreamSource(streamMeta.source);
    }
  }, [streamMeta?.source]);

  const handleModeChange = async (mode: StressMode) => {
    const previous = activeMode;
    setActiveMode(mode);
    setIsChanging(true);
    try {
      const response = await fetch("/api/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ mode }),
      });

      if (!response.ok) {
        throw new Error(`Failed to switch mode (${response.status})`);
      }
    } catch (error) {
      console.error("Failed to change stress mode:", error);
      setActiveMode(previous);
    } finally {
      setIsChanging(false);
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
      setPredictionError(null);
      return;
    }
    setIsPredicting(true);
    setPredictionError(null);

    try {
      const response = await fetch(DEFAULT_PREDICTION_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(featurePayload),
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(`Prediction request failed (${response.status})`);
      }

      const data = await response.json();
      const rawPrediction = (data?.prediction ??
        data?.result ??
        data) as unknown;

      let interpretedDetails: Record<string, unknown> | null = null;
      const maybeTrigger = (label: string) => {
        const isStressed = label.toLowerCase().includes("stress");
        const now = Date.now();
        const cooldownMs = 8000; // 8s cooldown to avoid spamming
        if (
          isStressed &&
          (!lastWasStressedRef.current ||
            now - lastTriggerRef.current > cooldownMs)
        ) {
          onStressTrigger?.(now);
          lastTriggerRef.current = now;
        }
        lastWasStressedRef.current = isStressed;
      };

      if (typeof rawPrediction === "number") {
        const label = rawPrediction === 1 ? "Stressed" : "Not stressed";
        setPredictionResult(label);
        maybeTrigger(label);
      } else if (typeof rawPrediction === "string") {
        setPredictionResult(rawPrediction);
        maybeTrigger(rawPrediction);
      } else if (rawPrediction && typeof rawPrediction === "object") {
        const nested = rawPrediction as Record<string, unknown>;
        if (typeof nested.prediction === "number") {
          const label = nested.prediction === 1 ? "Stressed" : "Not stressed";
          setPredictionResult(label);
          maybeTrigger(label);
        } else if (typeof nested.label === "string") {
          setPredictionResult(nested.label);
          maybeTrigger(String(nested.label));
        } else {
          setPredictionResult("See details below");
        }
        interpretedDetails = nested;
      } else {
        setPredictionResult("Prediction received");
      }

      if (!interpretedDetails && data && typeof data === "object") {
        interpretedDetails = data as Record<string, unknown>;
      }

      if (interpretedDetails) {
        setPredictionDetails(interpretedDetails);
      }
    } catch (error) {
      console.error("Failed to perform stress inference", error);
      setPredictionError(
        "Failed to reach the stress inference service. Ensure FastAPI is running."
      );
    } finally {
      setIsPredicting(false);
    }
  }, [featurePayload]);

  useEffect(() => {
    if (!featurePayload || monitoringPaused) {
      return;
    }

    const interval = setInterval(() => {
      if (!isPredicting) {
        void runInference();
      }
    }, 500);

    return () => clearInterval(interval);
  }, [featurePayload, isPredicting, runInference, monitoringPaused]);

  const updateStreamSource = useCallback(
    async (target: StreamSource) => {
      if (target === streamSource && streamMeta?.source === target) {
        return;
      }
      setPredictionError(null);
      const previous = streamSource;
      setStreamSource(target);
      setPredictionResult(null);
      setPredictionDetails(null);
      try {
        const response = await fetch("/api/stream", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ source: target }),
        });
        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          const errorMessage =
            typeof payload?.error === "string"
              ? payload.error
              : `Failed to switch to ${target.replace(/_/g, " ")}`;
          throw new Error(errorMessage);
        }
      } catch (error) {
        console.error("Failed to toggle stream source", error);
        setStreamSource(previous);
        setPredictionError("Unable to switch data source. Please try again.");
      }
    },
    [streamMeta?.source, streamSource]
  );

  const togglePrimarySource = useCallback(async () => {
    const next: StreamSource =
      streamSource === "simulated" ? "real_all" : "simulated";
    await updateStreamSource(next);
  }, [streamSource, updateStreamSource]);

  const realCounts = streamMeta?.availableRealSamples;

  const renderRealSourceSelector = () => (
    <div className="mt-4">
      <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">
        Real data slice
      </p>
      <div className="flex flex-wrap gap-2">
        {(
          ["real_all", "real_not_stressed", "real_stressed"] as Exclude<
            StreamSource,
            "simulated"
          >[]
        ).map((option) => {
          const label = REAL_SOURCE_LABELS[option];
          const count =
            option === "real_all"
              ? realCounts?.all
              : option === "real_not_stressed"
              ? realCounts?.notStressed
              : realCounts?.stressed;
          const disabled = typeof count === "number" ? count === 0 : false;
          const isActive = streamSource === option;
          return (
            <button
              key={option}
              type="button"
              onClick={() => updateStreamSource(option)}
              disabled={disabled}
              className={`
                px-3 py-2 text-xs font-semibold rounded-lg border transition
                ${
                  isActive
                    ? "border-indigo-500 bg-indigo-100 text-indigo-700"
                    : "border-slate-200 bg-white text-slate-600 hover:bg-slate-100"
                }
                ${disabled ? "opacity-40 cursor-not-allowed" : ""}
              `}
            >
              <div>{label}</div>
              {typeof count === "number" && (
                <div className="text-[10px] text-slate-400 mt-1">
                  {count} samples
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );

  const currentModeDisplay =
    streamSource === "simulated"
      ? activeLabel
      : describeRealSource(streamSource) ?? "Real Stream";

  return (
    <div className="bg-slate-100 rounded-2xl p-7 shadow-lg border border-slate-200 sticky top-8">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-900 mb-3">
          Stress Simulator
        </h3>
        <p className="text-slate-600 text-sm leading-relaxed">
          Toggle between neutral and stressed physiology
        </p>
        <button
          type="button"
          onClick={togglePrimarySource}
          className="mt-4 inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-100 transition"
        >
          {streamSource === "simulated"
            ? "Switch to Real Data Stream"
            : "Return to Simulated Stream"}
        </button>
        <p className="mt-2 text-xs uppercase tracking-wide text-slate-500">
          Current source:{" "}
          <span className="text-slate-800">
            {streamSource === "simulated"
              ? "Simulated"
              : describeRealSource(streamSource) ?? "Real Stream"}
          </span>
        </p>
        {streamSource !== "simulated" && renderRealSourceSelector()}
      </div>

      <div className="space-y-3">
        {streamSource === "simulated"
          ? stressStates.map((state) => (
              <button
                key={state.mode}
                onClick={() => handleModeChange(state.mode)}
                disabled={isChanging}
                className={`
                w-full p-4 rounded-lg transition-all duration-300
                ${
                  activeMode === state.mode
                    ? `bg-gradient-to-r ${state.color} shadow-lg scale-105`
                    : "bg-white hover:bg-slate-100 hover:scale-102 border border-slate-200"
                }
                ${
                  isChanging
                    ? "opacity-50 cursor-not-allowed"
                    : "cursor-pointer"
                }
                disabled:cursor-not-allowed
              `}
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`
                p-2 rounded-lg
                ${activeMode === state.mode ? "bg-white/20" : "bg-slate-200"}
              `}
                  >
                    {state.icon}
                  </div>
                  <div className="flex-1 text-left">
                    <div
                      className={`font-semibold ${
                        activeMode === state.mode
                          ? "text-white"
                          : "text-slate-800"
                      }`}
                    >
                      {state.label}
                    </div>
                    <div
                      className={`text-xs ${
                        activeMode === state.mode
                          ? "text-white/80"
                          : "text-slate-500"
                      }`}
                    >
                      {state.description}
                    </div>
                  </div>
                  {activeMode === state.mode && (
                    <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                  )}
                </div>
              </button>
            ))
          : null}
      </div>
      <div className="mt-7 p-5 bg-white rounded-xl border border-dashed border-slate-300">
        <h4 className="text-sm font-semibold text-slate-800 mb-3">
          Model Inference Preview
        </h4>
        <div className="space-y-4">
          <div className="flex items-center justify-between text-xs text-slate-500 bg-slate-100 border border-slate-200 px-3 py-2 rounded-lg">
            <span>Inference updates every 0.5 seconds</span>
            <span
              className={`inline-flex items-center gap-1 ${
                isPredicting ? "text-emerald-600" : "text-slate-400"
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isPredicting ? "bg-emerald-500 animate-pulse" : "bg-slate-400"
                }`}
              />
              {isPredicting ? "Updating" : "Idle"}
            </span>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-100 p-4 text-xs text-slate-700">
            <div className="font-semibold text-slate-800 mb-2">
              Latest feature snapshot
            </div>
            {featurePayload ? (
              <dl className="grid grid-cols-2 gap-2">
                {Object.entries(featurePayload).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <dt className="uppercase tracking-wide text-slate-500">
                      {key}
                    </dt>
                    <dd className="font-mono text-slate-800">{value}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <p className="text-slate-500 text-center">
                Awaiting live telemetry...
              </p>
            )}
          </div>
          {predictionResult && (
            <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4">
              <p className="text-sm font-semibold text-emerald-700">
                Prediction:{" "}
                <span className="text-emerald-900">{predictionResult}</span>
              </p>
              {predictionDetails && (
                <pre className="mt-3 text-xs text-slate-700 bg-emerald-100 rounded-lg p-3 overflow-x-auto">
                  {JSON.stringify(predictionDetails, null, 2)}
                </pre>
              )}
            </div>
          )}
          {predictionError && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
              {predictionError}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
