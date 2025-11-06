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
    normal: 'bg-emerald-400',
    warning: 'bg-amber-400',
    danger: 'bg-rose-400'
  };

  const trendColors = {
    up: 'text-rose-500',
    down: 'text-emerald-500',
    stable: 'text-slate-400'
  };

  return (
    <div className="bg-white rounded-lg p-4 shadow-md border border-slate-200 hover:border-slate-300 transition-all">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-slate-100 rounded-md text-slate-700">
            {icon}
          </div>
          <div>
            <p className="text-slate-600 text-xs font-semibold">{title}</p>
            {subtitle && <p className="text-slate-400 text-[11px]">{subtitle}</p>}
          </div>
        </div>
        <div className={`w-2 h-2 rounded-full ${statusColors[status]}`}></div>
      </div>
      
      <div className="flex items-end justify-between">
        <div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-slate-900">
              {typeof value === 'number' ? value.toFixed(1) : value}
            </span>
            {unit && <span className="text-slate-500 text-base">{unit}</span>}
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
