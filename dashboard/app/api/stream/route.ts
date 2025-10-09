// API route for streaming simulated wearable data using Server-Sent Events
export const dynamic = 'force-dynamic';

interface WearableData {
  timestamp: number;
  heartRate: number;
  stressLevel: number;
  spo2: number;
  temperature: number;
  steps: number;
  calories: number;
  sleepQuality: number;
  activityLevel: string;
}

type ActivityMode = 'idle' | 'jogging' | 'sprinting' | 'basketball' | 'sleeping';

// Activity profiles with target ranges
const activityProfiles: Record<ActivityMode, {
  heartRateTarget: number;
  heartRateVariance: number;
  stressTarget: number;
  stressVariance: number;
  tempTarget: number;
  stepsPerSecond: number;
  caloriesPerSecond: number;
  activityLabel: string;
}> = {
  idle: {
    heartRateTarget: 70,
    heartRateVariance: 3,
    stressTarget: 25,
    stressVariance: 5,
    tempTarget: 36.5,
    stepsPerSecond: 0,
    caloriesPerSecond: 0.02,
    activityLabel: 'resting'
  },
  jogging: {
    heartRateTarget: 130,
    heartRateVariance: 8,
    stressTarget: 45,
    stressVariance: 8,
    tempTarget: 37.2,
    stepsPerSecond: 2.5,
    caloriesPerSecond: 0.15,
    activityLabel: 'moderate'
  },
  sprinting: {
    heartRateTarget: 170,
    heartRateVariance: 10,
    stressTarget: 75,
    stressVariance: 10,
    tempTarget: 37.8,
    stepsPerSecond: 4,
    caloriesPerSecond: 0.3,
    activityLabel: 'intense'
  },
  basketball: {
    heartRateTarget: 145,
    heartRateVariance: 15,
    stressTarget: 60,
    stressVariance: 12,
    tempTarget: 37.5,
    stepsPerSecond: 3,
    caloriesPerSecond: 0.2,
    activityLabel: 'intense'
  },
  sleeping: {
    heartRateTarget: 55,
    heartRateVariance: 2,
    stressTarget: 10,
    stressVariance: 3,
    tempTarget: 36.2,
    stepsPerSecond: 0,
    caloriesPerSecond: 0.01,
    activityLabel: 'resting'
  }
};

// Global state for current activity mode and time scale
let currentActivityMode: ActivityMode = 'idle';
let currentTimeScale: number = 1; // 1x, 2x, 3x, 4x, 5x

// Simulate realistic wearable data based on activity mode
function generateWearableData(prevData: WearableData | undefined, mode: ActivityMode, timeScale: number = 1): WearableData {
  const now = Date.now();
  const profile = activityProfiles[mode];


  if (!prevData) {
    return {
      timestamp: now,
      heartRate: profile.heartRateTarget,
      stressLevel: profile.stressTarget,
      spo2: 98,
      temperature: profile.tempTarget,
      steps: 0,
      calories: 0,
      sleepQuality: 85,
      activityLevel: profile.activityLabel
    };
  }


  const deltaTime = 1.0;

  const heartRateDiff = profile.heartRateTarget - prevData.heartRate;
  const heartRateChange = (heartRateDiff * 0.1 + (Math.random() - 0.5) * profile.heartRateVariance) * deltaTime;
  const newHeartRate = Math.max(45, Math.min(185, prevData.heartRate + heartRateChange));


  const stressDiff = profile.stressTarget - prevData.stressLevel;
  const stressChange = (stressDiff * 0.1 + (Math.random() - 0.5) * profile.stressVariance) * deltaTime;
  const newStress = Math.max(0, Math.min(100, prevData.stressLevel + stressChange));


  const tempDiff = profile.tempTarget - prevData.temperature;
  const tempChange = (tempDiff * 0.05 + (Math.random() - 0.5) * 0.1) * deltaTime;
  const newTemp = Math.max(35.5, Math.min(37.8, prevData.temperature + tempChange));


  const spo2Base = mode === 'sprinting' || mode === 'basketball' ? 96 : 98;
  const spo2Change = (Math.random() - 0.5) * 1 * deltaTime;
  const newSpo2 = Math.max(94, Math.min(100, spo2Base + spo2Change));


  const sleepChange = (Math.random() - 0.5) * 0.1 * deltaTime;
  const newSleepQuality = Math.max(70, Math.min(95, prevData.sleepQuality + sleepChange));

  return {
    timestamp: now,
    heartRate: Math.round(newHeartRate),
    stressLevel: Math.round(newStress * 10) / 10,
    spo2: Math.round(newSpo2 * 10) / 10,
    temperature: Math.round(newTemp * 10) / 10,
    steps: prevData.steps + Math.floor(profile.stepsPerSecond * deltaTime),
    calories: prevData.calories + Math.round(profile.caloriesPerSecond * deltaTime * 10) / 10,
    sleepQuality: Math.round(newSleepQuality * 10) / 10,
    activityLevel: profile.activityLabel
  };
}

export async function GET(request: Request) {
  const encoder = new TextEncoder();
  let currentData = generateWearableData(undefined, currentActivityMode, currentTimeScale);
  let currentIntervalId: NodeJS.Timeout | null = null;
  let lastTimeScale = currentTimeScale;

  const stream = new ReadableStream({
    async start(controller) {
      const sendData = () => {
        currentData = generateWearableData(currentData, currentActivityMode, currentTimeScale);
        const data = `data: ${JSON.stringify(currentData)}\n\n`;
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
      'Connection': 'keep-alive',
    },
  });
}

// Endpoint to change activity mode
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { mode } = body;

    if (mode && mode in activityProfiles) {
      currentActivityMode = mode as ActivityMode;
      return Response.json({ success: true, mode: currentActivityMode });
    }

    return Response.json({ success: false, error: 'Invalid activity mode' }, { status: 400 });
  } catch (error) {
    return Response.json({ success: false, error: 'Invalid request' }, { status: 400 });
  }
}

// Endpoint to change time scale
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

