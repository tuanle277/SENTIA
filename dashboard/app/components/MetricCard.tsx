'use client';

import { ReactNode } from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  icon: ReactNode;
  trend?: 'up' | 'down' | 'stable';
  status?: 'normal' | 'warning' | 'danger';
  subtitle?: string;
}

export default function MetricCard({ 
  title, 
  value, 
  unit, 
  icon, 
  trend,
  status = 'normal',
  subtitle 
}: MetricCardProps) {
  const statusColors = {
    normal: 'bg-green-500',
    warning: 'bg-yellow-500',
    danger: 'bg-red-500'
  };

  const trendColors = {
    up: 'text-red-400',
    down: 'text-green-400',
    stable: 'text-gray-400'
  };

  return (
    <div className="bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-700 hover:border-gray-600 transition-all">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gray-700 rounded-lg">
            {icon}
          </div>
          <div>
            <p className="text-gray-400 text-sm font-medium">{title}</p>
            {subtitle && <p className="text-gray-500 text-xs">{subtitle}</p>}
          </div>
        </div>
        <div className={`w-2 h-2 rounded-full ${statusColors[status]}`}></div>
      </div>
      
      <div className="flex items-end justify-between">
        <div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-bold text-white">
              {typeof value === 'number' ? value.toFixed(1) : value}
            </span>
            {unit && <span className="text-gray-400 text-lg">{unit}</span>}
          </div>
        </div>
        
        {trend && (
          <div className={`text-sm ${trendColors[trend]} flex items-center gap-1`}>
            {trend === 'up' && '↑'}
            {trend === 'down' && '↓'}
            {trend === 'stable' && '→'}
          </div>
        )}
      </div>
    </div>
  );
}

