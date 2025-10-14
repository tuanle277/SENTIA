import { existsSync, readFileSync } from "fs";
import path from "path";

export const dynamic = "force-dynamic";

type FeatureKey = "hrvMeanNN" | "edaMean" | "accMagMean" | "tempMean";
type StressMode = "not_stressed" | "stressed";

interface FeatureStreamData {
  timestamp: number;
  hrvMeanNN: number;
  edaMean: number;
  accMagMean: number;
  tempMean: number;
}

interface FeatureStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  grad: number;
}

interface StressProfile {
  stats: Record<FeatureKey, FeatureStats>;
}

interface RunningStats {
  count: number;
  mean: number;
  m2: number;
  min: number;
  max: number;
  prev: number | null;
  gradSum: number;
  gradCount: number;
}

type StatsTable = Record<FeatureKey, RunningStats>;
type StressStats = Record<StressMode, StatsTable>;

const FEATURE_KEYS: FeatureKey[] = [
  "hrvMeanNN",
  "edaMean",
  "accMagMean",
  "tempMean",
];

const FALLBACK_STATS: Record<StressMode, Record<FeatureKey, FeatureStats>> = {
  not_stressed: {
    hrvMeanNN: {
      mean: 883.455722,
      std: 127.843995,
      min: 572.265625,
      max: 2022.321429,
      grad: 26.607029,
    },
    edaMean: {
      mean: 2.080201,
      std: 2.199223,
      min: 0,
      max: 9.017543,
      grad: 0.008954,
    },
    accMagMean: {
      mean: 44.143115,
      std: 29.234023,
      min: 0,
      max: 65.890039,
      grad: 0.035309,
    },
    tempMean: {
      mean: 23.451155,
      std: 15.562609,
      min: 0,
      max: 35.899333,
      grad: 0.008853,
    },
  },
  stressed: {
    hrvMeanNN: {
      mean: 835.74555,
      std: 164.16381,
      min: 438.964844,
      max: 2011.160714,
      grad: 34.16519,
    },
    edaMean: {
      mean: 3.711731,
      std: 3.752728,
      min: 0,
      max: 15.695068,
      grad: 0.012802,
    },
    accMagMean: {
      mean: 61.600711,
      std: 11.4306,
      min: 0,
      max: 68.191666,
      grad: 0.057883,
    },
    tempMean: {
      mean: 31.560332,
      std: 5.918486,
      min: 0,
      max: 34.546,
      grad: 0.007294,
    },
  },
};

function createRunningStats(): RunningStats {
  return {
    count: 0,
    mean: 0,
    m2: 0,
    min: Number.POSITIVE_INFINITY,
    max: Number.NEGATIVE_INFINITY,
    prev: null,
    gradSum: 0,
    gradCount: 0,
  };
}

function initialiseStatsTable(): StatsTable {
  return {
    hrvMeanNN: createRunningStats(),
    edaMean: createRunningStats(),
    accMagMean: createRunningStats(),
    tempMean: createRunningStats(),
  };
}

function datasetPath(): string {
  return path.resolve(
    process.cwd(),
    "..",
    "data",
    "processed",
    "merged_features_dataset.csv"
  );
}

const CSV_COLUMN_MAP: Record<FeatureKey, string> = {
  hrvMeanNN: "HRV_MeanNN",
  edaMean: "EDA_Mean",
  accMagMean: "ACC_Mag_Mean",
  tempMean: "TEMP_Mean",
};

// Align validation with the same predictor used by the UI
// Prefer local predictor by default to avoid remote TLS issues during dev
const PREDICT_ENDPOINT =
  process.env.NEXT_PUBLIC_PREDICT_ENDPOINT || "http://localhost:5000/predict";

function toPredictorPayload(sample: FeatureStreamData): Record<string, number> {
  return {
    HRV_MeanNN: Number(sample.hrvMeanNN.toFixed(3)),
    EDA_Mean: Number(sample.edaMean.toFixed(3)),
    ACC_Mag_Mean: Number(sample.accMagMean.toFixed(3)),
    TEMP_Mean: Number(sample.tempMean.toFixed(3)),
  };
}

function extractConfidenceFromResponse(json: unknown): number | null {
  if (json == null) return null;
  const data = json as Record<string, unknown>;
  const nested =
    (data.result as Record<string, unknown> | undefined) ?? undefined;
  const candidates = [
    data.probability,
    data.prob,
    data.confidence,
    nested?.probability,
    nested?.prob,
  ];
  for (const c of candidates) {
    if (typeof c === "number" && Number.isFinite(c)) return c;
  }
  const pred = (data.prediction ?? data.label) as unknown;
  if (typeof pred === "number") return pred === 1 ? 0.99 : 0.01;
  if (typeof pred === "string") {
    const s = pred.toLowerCase();
    if (s.includes("stress")) return 0.99;
    if (s.includes("not")) return 0.01;
  }
  return null;
}

function loadStressProfiles(): Record<StressMode, StressProfile> {
  const csvPath = datasetPath();

  if (!existsSync(csvPath)) {
    console.warn("[stream api] Dataset not found, using fallback statistics.");
    return {
      not_stressed: { stats: FALLBACK_STATS.not_stressed },
      stressed: { stats: FALLBACK_STATS.stressed },
    };
  }

  try {
    const raw = readFileSync(csvPath, "utf-8");
    const lines = raw.split(/\r?\n/).filter((line) => line.trim().length > 0);

    if (lines.length < 2) {
      console.warn("[stream api] Dataset empty, using fallback statistics.");
      return {
        not_stressed: { stats: FALLBACK_STATS.not_stressed },
        stressed: { stats: FALLBACK_STATS.stressed },
      };
    }

    const header = lines[0].split(",");
    const indexMap: Record<string, number> = {};
    for (const col of header) {
      indexMap[col] = indexMap[col] ?? header.indexOf(col);
    }

    const labelIndex = indexMap["stress_label"];
    if (labelIndex === undefined) {
      console.warn(
        "[stream api] stress_label column missing, using fallback statistics."
      );
      return {
        not_stressed: { stats: FALLBACK_STATS.not_stressed },
        stressed: { stats: FALLBACK_STATS.stressed },
      };
    }

    const stats: StressStats = {
      not_stressed: initialiseStatsTable(),
      stressed: initialiseStatsTable(),
    };

    for (let i = 1; i < lines.length; i += 1) {
      const row = lines[i];
      if (!row) continue;

      const cells = row.split(",");
      const labelValue = Number.parseInt(cells[labelIndex], 10);
      if (labelValue !== 0 && labelValue !== 1) {
        continue;
      }
      const mode: StressMode = labelValue === 0 ? "not_stressed" : "stressed";
      const table = stats[mode];

      for (const key of FEATURE_KEYS) {
        const columnName = CSV_COLUMN_MAP[key];
        const idx = indexMap[columnName];
        if (idx === undefined) {
          continue;
        }
        const cell = cells[idx];
        const value = Number.parseFloat(cell);
        if (!Number.isFinite(value)) continue;

        const running = table[key];
        running.count += 1;
        const delta = value - running.mean;
        running.mean += delta / running.count;
        const delta2 = value - running.mean;
        running.m2 += delta * delta2;
        running.min = Math.min(running.min, value);
        running.max = Math.max(running.max, value);
        if (running.prev !== null) {
          running.gradSum += Math.abs(value - running.prev);
          running.gradCount += 1;
        }
        running.prev = value;
      }
    }

    const profiles: Record<StressMode, StressProfile> = {
      not_stressed: { stats: {} as Record<FeatureKey, FeatureStats> },
      stressed: { stats: {} as Record<FeatureKey, FeatureStats> },
    };

    for (const mode of Object.keys(stats) as StressMode[]) {
      const table = stats[mode];
      const profileStats: Record<FeatureKey, FeatureStats> = {} as Record<
        FeatureKey,
        FeatureStats
      >;

      for (const key of FEATURE_KEYS) {
        const running = table[key];
        if (running.count === 0) {
          console.warn(
            `[stream api] No data for ${mode} ${key}, using fallback statistics.`
          );
          profileStats[key] = FALLBACK_STATS[mode][key];
          continue;
        }

        const mean = running.mean;
        const variance = running.m2 / running.count;
        const std = Math.sqrt(Math.max(variance, 0));
        const grad =
          running.gradCount > 0
            ? running.gradSum / running.gradCount
            : FALLBACK_STATS[mode][key].grad;

        profileStats[key] = {
          mean,
          std,
          min: running.min,
          max: running.max,
          grad,
        };
      }

      profiles[mode] = { stats: profileStats };
    }

    return profiles;
  } catch (error) {
    console.error(
      "[stream api] Failed to parse dataset, using fallback statistics.",
      error
    );
    return {
      not_stressed: { stats: FALLBACK_STATS.not_stressed },
      stressed: { stats: FALLBACK_STATS.stressed },
    };
  }
}

const stressProfiles = loadStressProfiles();
let validationEnabled = process.env.STRESS_VALIDATION_ENABLED !== "false";

// Feature separation directions: values tend to be higher in stressed (+1) or lower (-1)
const FEATURE_DIRECTION: Record<FeatureKey, 1 | -1> = {
  hrvMeanNN: -1,
  edaMean: 1,
  accMagMean: 1,
  tempMean: 1,
};

function getBiasedTarget(
  stats: FeatureStats,
  key: FeatureKey,
  mode: StressMode
): number {
  const direction = FEATURE_DIRECTION[key];
  const std = stats.std > 0 ? stats.std : stats.grad || 1;
  // Stronger margin toward class-distinctive side for stressed, milder for not_stressed
  const marginMultiplier = mode === "stressed" ? 0.8 : 0.35;
  const signedMargin = direction * std * marginMultiplier;
  return stats.mean + signedMargin;
}

function clampAroundTarget(
  value: number,
  stats: FeatureStats,
  target: number
): number {
  const std = stats.std > 0 ? stats.std : stats.grad || 1;
  const halfRange = std * 1.2; // narrow band around biased target
  const lower = target - halfRange;
  const upper = target + halfRange;
  return Math.max(lower, Math.min(upper, value));
}

function randomNormal(mean = 0, std = 1): number {
  if (!Number.isFinite(std) || std <= 0) return mean;
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return (
    mean + Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * std
  );
}

function clampValue(value: number, stats: FeatureStats): number {
  const padding = stats.std || stats.grad || 1;
  const lower = Math.min(stats.min, stats.max) - padding;
  const upper = Math.max(stats.min, stats.max) + padding;
  if (!Number.isFinite(lower) || !Number.isFinite(upper)) {
    return value;
  }
  return Math.max(lower, Math.min(upper, value));
}

function initialiseData(mode: StressMode): FeatureStreamData {
  const profile = stressProfiles[mode];
  const result: Partial<FeatureStreamData> = { timestamp: Date.now() };
  for (const key of FEATURE_KEYS) {
    const stats = profile.stats[key];
    const std = stats.std > 0 ? stats.std : stats.grad || 1;
    const target = getBiasedTarget(stats, key, mode);
    const sampled = randomNormal(target, std * 0.25);
    const clamped = clampAroundTarget(sampled, stats, target);
    result[key] = Number(clamped.toFixed(key === "hrvMeanNN" ? 0 : 3));
  }
  return result as FeatureStreamData;
}

function stepFeature(
  previous: number,
  stats: FeatureStats,
  key: FeatureKey,
  mode: StressMode
): number {
  const smoothing = 0.3;
  const baseStd = stats.std > 0 ? stats.std : stats.grad || 1;
  const target = getBiasedTarget(stats, key, mode);
  const gradientNoise = stats.grad ? stats.grad * 0.35 : baseStd * 0.035;
  const noise = randomNormal(0, gradientNoise);
  const next = previous + (target - previous) * smoothing + noise;
  return clampAroundTarget(next, stats, target);
}

async function generateFeatureData(
  previous: FeatureStreamData | undefined,
  mode: StressMode
): Promise<FeatureStreamData> {
  if (!previous) {
    return initialiseData(mode);
  }

  const profile = stressProfiles[mode];

  // Function to generate a single feature data object
  const generateSingleFeatureData = (): FeatureStreamData => ({
    timestamp: Date.now(),
    hrvMeanNN: Number(
      stepFeature(
        previous.hrvMeanNN,
        profile.stats.hrvMeanNN,
        "hrvMeanNN",
        mode
      ).toFixed(0)
    ),
    edaMean: Number(
      stepFeature(
        previous.edaMean,
        profile.stats.edaMean,
        "edaMean",
        mode
      ).toFixed(3)
    ),
    accMagMean: Number(
      stepFeature(
        previous.accMagMean,
        profile.stats.accMagMean,
        "accMagMean",
        mode
      ).toFixed(3)
    ),
    tempMean: Number(
      stepFeature(
        previous.tempMean,
        profile.stats.tempMean,
        "tempMean",
        mode
      ).toFixed(3)
    ),
  });

  let featureData: FeatureStreamData;
  let isValid = false;
  let attempts = 0;
  const validationThreshold = 0.8;
  const maxValidationAttempts = 8;
  const requestTimeoutMs = 1500;
  // Circuit breaker for remote predictor
  // If remote endpoint fails once, skip trying it for a cooldown period
  const remoteCooldownMs = 5 * 60 * 1000;
  let staticRemoteBlockedUntil = (generateFeatureData as any)
    .remoteBlockedUntil as number | undefined;
  if (typeof staticRemoteBlockedUntil !== "number") {
    staticRemoteBlockedUntil = 0;
  }
  const nowTs = Date.now();

  // If validation disabled, just return one generated sample
  if (!validationEnabled) {
    return generateSingleFeatureData();
  }

  // Keep generating feature data until the confidence is high enough
  while (!isValid && attempts < maxValidationAttempts) {
    featureData = generateSingleFeatureData();
    console.log("Checking confidence with data:", featureData);

    // Call the /check API endpoint to validate the confidence
    try {
      const abortController = new AbortController();
      const timeoutId = setTimeout(
        () => abortController.abort(),
        requestTimeoutMs
      );

      // try primary predictor first (skipped during cooldown)
      let response: Response | undefined;
      if (nowTs >= staticRemoteBlockedUntil) {
        response = await fetch(PREDICT_ENDPOINT, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(toPredictorPayload(featureData)),
          signal: abortController.signal,
        });
      }
      // if remote skipped or failed, response may be undefined; try local as fallback
      if (!response || !response.ok) {
        try {
          response = await fetch("http://localhost:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(toPredictorPayload(featureData)),
            signal: abortController.signal,
          });
        } catch {}
      }

      console.log("Checking confidence with data:", featureData, response);

      if (!response.ok) {
        console.error("Error checking confidence:", await response.text());
        throw new Error("Failed to check confidence");
      }

      const result = await response.json();
      const confidence = extractConfidenceFromResponse(result);
      isValid = confidence !== null ? confidence >= validationThreshold : true;
      clearTimeout(timeoutId);
    } catch (error) {
      // Block remote endpoint for a cooldown window on network failure
      (generateFeatureData as any).remoteBlockedUntil =
        Date.now() + remoteCooldownMs;
      // Throttle noisy logs
      if (attempts === 0) {
        console.warn(
          "Confidence check failed or timed out; proceeding with generated data.",
          (error as Error)?.message ?? error
        );
      }
      // Break out to avoid stalling the stream if the checker is down
      isValid = true;
    }
    attempts += 1;
  }

  return featureData!;
}

let currentStressMode: StressMode = "not_stressed";
let currentTimeScale = 1;

export async function GET(request: Request) {
  const encoder = new TextEncoder();
  let currentData = generateFeatureData(undefined, currentStressMode);
  let currentIntervalId: NodeJS.Timeout | null = null;
  let lastTimeScale = currentTimeScale;

  const stream = new ReadableStream({
    async start(controller) {
      const sendData = async () => {
        try {
          console.log("[stream api] Generating new feature data...");
          // notify clients that generation is in progress
          controller.enqueue(
            encoder.encode(
              `event: status\ndata: ${JSON.stringify({ generating: true })}\n\n`
            )
          );
          const resolved = await currentData; // await the promise to get concrete data
          const data = `data: ${JSON.stringify(resolved)}\n\n`;
          controller.enqueue(encoder.encode(data));
          // prepare next tick's promise
          currentData = generateFeatureData(resolved, currentStressMode);
          // notify clients generation is done for this tick
          controller.enqueue(
            encoder.encode(
              `event: status\ndata: ${JSON.stringify({
                generating: false,
              })}\n\n`
            )
          );
        } catch (err) {
          console.error(
            "[stream api] Failed to generate or send data; using fallback.",
            err
          );
          const fallback = initialiseData(currentStressMode);
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify(fallback)}\n\n`)
          );
          currentData = generateFeatureData(fallback, currentStressMode);
        }
      };

      const updateInterval = () => {
        if (currentIntervalId) {
          clearInterval(currentIntervalId);
        }

        const intervalMs = 1000 / currentTimeScale;
        currentIntervalId = setInterval(sendData, intervalMs);
        lastTimeScale = currentTimeScale;
      };

      sendData();
      updateInterval();

      const checkInterval = setInterval(() => {
        if (currentTimeScale !== lastTimeScale) {
          updateInterval();
        }
      }, 100);

      request.signal.addEventListener("abort", () => {
        if (currentIntervalId) clearInterval(currentIntervalId);
        clearInterval(checkInterval);
        controller.close();
      });
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { mode } = body;

    if (mode && mode in stressProfiles) {
      currentStressMode = mode as StressMode;
      return Response.json({ success: true, mode: currentStressMode });
    }

    return Response.json(
      { success: false, error: "Invalid stress mode" },
      { status: 400 }
    );
  } catch (error) {
    return Response.json(
      { success: false, error: "Invalid request" },
      { status: 400 }
    );
  }
}

export async function PATCH(request: Request) {
  try {
    const body = await request.json();
    const { timeScale } = body;

    if (
      timeScale &&
      typeof timeScale === "number" &&
      timeScale >= 1 &&
      timeScale <= 5
    ) {
      currentTimeScale = timeScale;
      return Response.json({ success: true, timeScale: currentTimeScale });
    }

    return Response.json(
      { success: false, error: "Invalid time scale (must be 1-5)" },
      { status: 400 }
    );
  } catch (error) {
    return Response.json(
      { success: false, error: "Invalid request" },
      { status: 400 }
    );
  }
}
