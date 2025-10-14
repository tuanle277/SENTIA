import { existsSync, readFileSync } from 'fs';
import path from 'path';

export const dynamic = 'force-dynamic';

type FeatureKey = 'hrvMeanNN' | 'edaMean' | 'accMagMean' | 'tempMean';
type StressMode = 'not_stressed' | 'stressed';

interface FeatureStreamData {
  timestamp: number;
  hrvMeanNN: number;
  edaMean: number;
  accMagMean: number;
  tempMean: number;
  stressLabel?: number;
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
interface RealDataRecord {
  hrvMeanNN: number;
  edaMean: number;
  accMagMean: number;
  tempMean: number;
  stressLabel: number;
}

const FEATURE_KEYS: FeatureKey[] = ['hrvMeanNN', 'edaMean', 'accMagMean', 'tempMean'];

const FALLBACK_STATS: Record<StressMode, Record<FeatureKey, FeatureStats>> = {
  not_stressed: {
    hrvMeanNN: { mean: 883.455722, std: 127.843995, min: 572.265625, max: 2022.321429, grad: 26.607029 },
    edaMean: { mean: 2.080201, std: 2.199223, min: 0, max: 9.017543, grad: 0.008954 },
    accMagMean: { mean: 44.143115, std: 29.234023, min: 0, max: 65.890039, grad: 0.035309 },
    tempMean: { mean: 23.451155, std: 15.562609, min: 0, max: 35.899333, grad: 0.008853 },
  },
  stressed: {
    hrvMeanNN: { mean: 835.74555, std: 164.16381, min: 438.964844, max: 2011.160714, grad: 34.16519 },
    edaMean: { mean: 3.711731, std: 3.752728, min: 0, max: 15.695068, grad: 0.012802 },
    accMagMean: { mean: 61.600711, std: 11.4306, min: 0, max: 68.191666, grad: 0.057883 },
    tempMean: { mean: 31.560332, std: 5.918486, min: 0, max: 34.546, grad: 0.007294 },
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
  return path.resolve(process.cwd(), '..', 'data', 'processed', 'merged_features_dataset.csv');
}

const CSV_COLUMN_MAP: Record<FeatureKey, string> = {
  hrvMeanNN: 'HRV_MeanNN',
  edaMean: 'EDA_Mean',
  accMagMean: 'ACC_Mag_Mean',
  tempMean: 'TEMP_Mean',
};
const REAL_DATASET_FILE = path.resolve(
  process.cwd(),
  '..',
  'data',
  'processed',
  'stream_features_dataset.csv',
);

function loadRealDataset(): RealDataRecord[] {
  if (!existsSync(REAL_DATASET_FILE)) {
    console.warn('[stream api] stream_features_dataset.csv not found; real streaming disabled.');
    return [];
  }

  try {
    const raw = readFileSync(REAL_DATASET_FILE, 'utf-8');
    const lines = raw.split(/\r?\n/).filter((line) => line.trim().length > 0);
    if (lines.length < 2) {
      console.warn('[stream api] stream_features_dataset.csv empty.');
      return [];
    }

    const header = lines[0].split(',');
    const indexMap: Record<string, number> = {};
    header.forEach((col, idx) => {
      indexMap[col] = idx;
    });
    const required = ['HRV_MeanNN', 'EDA_Mean', 'ACC_Mag_Mean', 'TEMP_Mean', 'stress_label'];
    if (required.some((col) => indexMap[col] === undefined)) {
      console.warn('[stream api] stream_features_dataset.csv missing required columns.');
      return [];
    }

    const records: RealDataRecord[] = [];
    for (let i = 1; i < lines.length; i += 1) {
      const row = lines[i];
      if (!row) continue;
      const cells = row.split(',');
      const record: RealDataRecord = {
        hrvMeanNN: Number.parseFloat(cells[indexMap['HRV_MeanNN']]),
        edaMean: Number.parseFloat(cells[indexMap['EDA_Mean']]),
        accMagMean: Number.parseFloat(cells[indexMap['ACC_Mag_Mean']]),
        tempMean: Number.parseFloat(cells[indexMap['TEMP_Mean']]),
        stressLabel: Number.parseInt(cells[indexMap['stress_label']], 10) || 0,
      };
      if (
        Number.isFinite(record.hrvMeanNN) &&
        Number.isFinite(record.edaMean) &&
        Number.isFinite(record.accMagMean) &&
        Number.isFinite(record.tempMean)
      ) {
        records.push(record);
      }
    }

    return records;
  } catch (error) {
    console.error('[stream api] Failed to load stream_features_dataset.csv', error);
    return [];
  }
}

function loadStressProfiles(): Record<StressMode, StressProfile> {
  const csvPath = datasetPath();

  if (!existsSync(csvPath)) {
    console.warn('[stream api] Dataset not found, using fallback statistics.');
    return {
      not_stressed: { stats: FALLBACK_STATS.not_stressed },
      stressed: { stats: FALLBACK_STATS.stressed },
    };
  }

  try {
    const raw = readFileSync(csvPath, 'utf-8');
    const lines = raw.split(/\r?\n/).filter((line) => line.trim().length > 0);

    if (lines.length < 2) {
      console.warn('[stream api] Dataset empty, using fallback statistics.');
      return {
        not_stressed: { stats: FALLBACK_STATS.not_stressed },
        stressed: { stats: FALLBACK_STATS.stressed },
      };
    }

    const header = lines[0].split(',');
    const indexMap: Record<string, number> = {};
    for (const col of header) {
      indexMap[col] = indexMap[col] ?? header.indexOf(col);
    }

    const labelIndex = indexMap['stress_label'];
    if (labelIndex === undefined) {
      console.warn('[stream api] stress_label column missing, using fallback statistics.');
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

      const cells = row.split(',');
      const labelValue = Number.parseInt(cells[labelIndex], 10);
      if (labelValue !== 0 && labelValue !== 1) {
        continue;
      }
      const mode: StressMode = labelValue === 0 ? 'not_stressed' : 'stressed';
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
      const profileStats: Record<FeatureKey, FeatureStats> = {} as Record<FeatureKey, FeatureStats>;

      for (const key of FEATURE_KEYS) {
        const running = table[key];
        if (running.count === 0) {
          console.warn(`[stream api] No data for ${mode} ${key}, using fallback statistics.`);
          profileStats[key] = FALLBACK_STATS[mode][key];
          continue;
        }

        const mean = running.mean;
        const variance = running.m2 / running.count;
        const std = Math.sqrt(Math.max(variance, 0));
        const grad =
          running.gradCount > 0 ? running.gradSum / running.gradCount : FALLBACK_STATS[mode][key].grad;

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
    console.error('[stream api] Failed to parse dataset, using fallback statistics.', error);
    return {
      not_stressed: { stats: FALLBACK_STATS.not_stressed },
      stressed: { stats: FALLBACK_STATS.stressed },
    };
  }
}

const stressProfiles = loadStressProfiles();
const realDataRecords = loadRealDataset();

function randomNormal(mean = 0, std = 1): number {
  if (!Number.isFinite(std) || std <= 0) return mean;
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return mean + Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * std;
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
    const baselineStd = stats.std > 0 ? stats.std : stats.grad || 1;
    const sampled = randomNormal(stats.mean, baselineStd * 0.5);
    result[key] = Number(clampValue(sampled, stats).toFixed(key === 'hrvMeanNN' ? 0 : 3));
  }
  return result as FeatureStreamData;
}

function stepFeature(previous: number, stats: FeatureStats): number {
  const smoothing = 0.15;
  const target = stats.mean;
  const baseStd = stats.std > 0 ? stats.std : stats.grad || 1;
  const gradientNoise = stats.grad || baseStd * 0.1;
  const noise = randomNormal(0, gradientNoise);
  const next = previous + (target - previous) * smoothing + noise;
  return clampValue(next, stats);
}

function generateFeatureData(
  previous: FeatureStreamData | undefined,
  mode: StressMode,
): FeatureStreamData {
  if (!previous) {
    return initialiseData(mode);
  }
  const profile = stressProfiles[mode];
  return {
    timestamp: Date.now(),
    hrvMeanNN: Number(stepFeature(previous.hrvMeanNN, profile.stats.hrvMeanNN).toFixed(0)),
    edaMean: Number(stepFeature(previous.edaMean, profile.stats.edaMean).toFixed(3)),
    accMagMean: Number(stepFeature(previous.accMagMean, profile.stats.accMagMean).toFixed(3)),
    tempMean: Number(stepFeature(previous.tempMean, profile.stats.tempMean).toFixed(3)),
  };
}

let currentStressMode: StressMode = 'not_stressed';
let currentTimeScale = 1;
let streamingSource: 'simulated' | 'real' = 'simulated';
let realDataIndex = 0;
let simulatedData: FeatureStreamData | undefined;

function getNextRealData(): FeatureStreamData {
  if (!realDataRecords.length) {
    return generateFeatureData(undefined, currentStressMode);
  }
  const record = realDataRecords[realDataIndex];
  realDataIndex = (realDataIndex + 1) % realDataRecords.length;
  return {
    timestamp: Date.now(),
    hrvMeanNN: Number(record.hrvMeanNN.toFixed(3)),
    edaMean: Number(record.edaMean.toFixed(3)),
    accMagMean: Number(record.accMagMean.toFixed(3)),
    tempMean: Number(record.tempMean.toFixed(3)),
    stressLabel: record.stressLabel,
  };
}

export async function GET(request: Request) {
  const encoder = new TextEncoder();
  let currentIntervalId: NodeJS.Timeout | null = null;
  let lastTimeScale = currentTimeScale;

  const stream = new ReadableStream({
    async start(controller) {
      const sendData = () => {
        if (streamingSource === 'real') {
          simulatedData = undefined;
        } else {
          simulatedData = generateFeatureData(simulatedData, currentStressMode);
        }
        const payload =
          streamingSource === 'real'
            ? getNextRealData()
            : (simulatedData ?? generateFeatureData(undefined, currentStressMode));
        const data = `data: ${JSON.stringify(payload)}\n\n`;
        controller.enqueue(encoder.encode(data));
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

      request.signal.addEventListener('abort', () => {
        if (currentIntervalId) clearInterval(currentIntervalId);
        clearInterval(checkInterval);
        controller.close();
      });
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { mode, source } = body as { mode?: StressMode; source?: 'simulated' | 'real' };

    if (source && (source === 'simulated' || source === 'real')) {
      streamingSource = source;
      if (streamingSource === 'real') {
        realDataIndex = 0;
      }
      return Response.json({ success: true, source: streamingSource });
    }

    if (mode && mode in stressProfiles) {
      currentStressMode = mode as StressMode;
      return Response.json({ success: true, mode: currentStressMode });
    }

    return Response.json({ success: false, error: 'Invalid stress mode' }, { status: 400 });
  } catch (error) {
    return Response.json({ success: false, error: 'Invalid request' }, { status: 400 });
  }
}

export async function PATCH(request: Request) {
  try {
    const body = await request.json();
    const { timeScale } = body;

    if (timeScale && typeof timeScale === 'number' && timeScale >= 1 && timeScale <= 5) {
      currentTimeScale = timeScale;
      return Response.json({ success: true, timeScale: currentTimeScale });
    }

    return Response.json({ success: false, error: 'Invalid time scale (must be 1-5)' }, { status: 400 });
  } catch (error) {
    return Response.json({ success: false, error: 'Invalid request' }, { status: 400 });
  }
}
