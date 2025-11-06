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
  compact?: boolean;
}

export default function MetricCard({ 
  title, 
  value, 
  unit, 
  icon, 
  trend,
  status = 'normal',
  subtitle,
  compact = false
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
    <div className={`bg-white rounded-xl shadow-md border border-slate-200 hover:border-slate-300 transition-all ${compact ? 'p-4' : 'p-6'}`}>
      <div className={`flex items-start justify-between ${compact ? 'mb-3' : 'mb-4'}`}>
        <div className={`flex items-center ${compact ? 'gap-2' : 'gap-3'}`}>
          <div className={`bg-slate-100 rounded-lg text-slate-700 ${compact ? 'p-1.5' : 'p-2'}`}>
            {icon}
          </div>
          <div>
            <p className={`text-slate-500 font-medium ${compact ? 'text-xs' : 'text-sm'}`}>{title}</p>
            {subtitle && <p className={`text-slate-400 ${compact ? 'text-[10px]' : 'text-xs'}`}>{subtitle}</p>}
          </div>
        </div>
        <div className={`rounded-full ${statusColors[status]} ${compact ? 'w-1.5 h-1.5' : 'w-2 h-2'}`}></div>
      </div>
      
      <div className="flex items-end justify-between">
        <div>
          <div className={`flex items-baseline ${compact ? 'gap-1' : 'gap-2'}`}>
            <span className={`font-bold text-slate-900 ${compact ? 'text-xl' : 'text-3xl'}`}>
              {typeof value === 'number' ? value.toFixed(1) : value}
            </span>
            {unit && <span className={`text-slate-500 ${compact ? 'text-sm' : 'text-lg'}`}>{unit}</span>}
          </div>
        </div>
        
        {trend && (
          <div className={`${compact ? 'text-xs' : 'text-sm'} ${trendColors[trend]} flex items-center gap-1`}>
            {trend === 'up' && '↑'}
            {trend === 'down' && '↓'}
            {trend === 'stable' && '→'}
          </div>
        )}
      </div>
    </div>
  );
}
