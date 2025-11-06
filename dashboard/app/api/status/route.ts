import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

interface SystemStatus {
  status: 'running' | 'offline';
  pending_notifications: number;
  last_update: number;
}

// Simple in-memory store to track notifications count
// In production, this would be a shared database or cache
let notificationCount = 0;

export async function GET() {
  // In production, this would check actual system status
  // For now, simulate a running system
  const status: SystemStatus = {
    status: 'running',
    pending_notifications: notificationCount,
    last_update: Date.now(),
  };

  return NextResponse.json(status);
}

// Export function to update notification count (used by notifications route)
export function updateNotificationCount(count: number) {
  notificationCount = count;
}

