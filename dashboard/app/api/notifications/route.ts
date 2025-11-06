import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

interface Notification {
  id: string;
  type: 'stress_detected' | 'info' | 'warning';
  message: string;
  timestamp: number;
  acknowledged?: boolean;
}

// In-memory store (in production, use a database)
let notifications: Notification[] = [];
let notificationIdCounter = 1;
let stressSimulationInterval: NodeJS.Timeout | null = null;

// Initialize stress simulation only on server side
function initializeStressSimulation() {
  if (typeof globalThis !== 'undefined' && !stressSimulationInterval) {
    stressSimulationInterval = setInterval(() => {
      // Randomly add stress notifications (for testing)
      if (Math.random() > 0.95) {
        const notification: Notification = {
          id: `stress-${notificationIdCounter++}`,
          type: 'stress_detected',
          message: 'Elevated stress levels detected. Your physiological indicators suggest you might be experiencing stress.',
          timestamp: Date.now(),
          acknowledged: false,
        };
        notifications.push(notification);
      }
    }, 30000); // Check every 30 seconds
  }
}

// Initialize on module load
if (typeof globalThis !== 'undefined') {
  initializeStressSimulation();
}

export async function GET() {
  const unacknowledged = notifications.filter((n) => !n.acknowledged);
  return NextResponse.json({ notifications: unacknowledged });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { action, id } = body;

    if (action === 'acknowledge' && id) {
      const notification = notifications.find((n) => n.id === id);
      if (notification) {
        notification.acknowledged = true;
      }
      const unacknowledged = notifications.filter((n) => !n.acknowledged);
      return NextResponse.json({ success: true, pending_count: unacknowledged.length });
    }

    if (action === 'clear') {
      notifications = [];
      return NextResponse.json({ success: true, pending_count: 0 });
    }

    if (action === 'simulate_stress') {
      const notification: Notification = {
        id: `stress-${notificationIdCounter++}`,
        type: 'stress_detected',
        message: 'Elevated stress levels detected. Your physiological indicators suggest you might be experiencing stress.',
        timestamp: Date.now(),
        acknowledged: false,
      };
      notifications.push(notification);
      return NextResponse.json({ success: true, notification });
    }

    return NextResponse.json({ success: false, error: 'Invalid action' }, { status: 400 });
  } catch (error) {
    return NextResponse.json({ success: false, error: 'Invalid request' }, { status: 400 });
  }
}

