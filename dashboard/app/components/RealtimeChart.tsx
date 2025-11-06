'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { format } from 'date-fns';

interface ChartDataPoint {
  timestamp: number;
  value: number;
}

interface RealtimeChartProps {
  data: ChartDataPoint[];
  title: string;
  color: string;
  unit?: string;
  domain?: [number, number];
  compact?: boolean;
}

export default function RealtimeChart({ data, title, color, unit, domain, compact = false }: RealtimeChartProps) {
  return (
    <div className={`bg-white rounded-xl shadow-md border border-slate-200 ${compact ? 'p-4' : 'p-6'}`}>
      <h3 className={`font-semibold text-slate-900 ${compact ? 'text-sm mb-3' : 'text-lg mb-4'}`}>{title}</h3>
      <ResponsiveContainer width="100%" height={compact ? 180 : 250}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id={`gradient-${color}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.3}/>
              <stop offset="95%" stopColor={color} stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
          <XAxis 
            dataKey="timestamp" 
            tickFormatter={(timestamp) => format(new Date(timestamp), 'HH:mm:ss')}
            stroke="#94A3B8"
            style={{ fontSize: compact ? '10px' : '12px' }}
          />
          <YAxis 
            stroke="#94A3B8"
            style={{ fontSize: compact ? '10px' : '12px' }}
            domain={domain || ['auto', 'auto']}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#F8FAFC', 
              border: '1px solid #E2E8F0',
              borderRadius: '8px',
              color: '#0F172A'
            }}
            labelFormatter={(timestamp) => format(new Date(timestamp), 'HH:mm:ss')}
            formatter={(value: number) => [`${value.toFixed(1)}${unit || ''}`, title]}
          />
          <Area 
            type="monotone" 
            dataKey="value" 
            stroke={color} 
            strokeWidth={2}
            fill={`url(#gradient-${color})`}
            animationDuration={300}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
