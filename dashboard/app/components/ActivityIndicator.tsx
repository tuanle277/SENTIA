'use client';

interface ActivityIndicatorProps {
  level: 'resting' | 'light' | 'moderate' | 'intense';
}

export default function ActivityIndicator({ level }: ActivityIndicatorProps) {
  const levels = {
    resting: { color: 'bg-blue-500', text: 'Resting', bars: 1 },
    light: { color: 'bg-green-500', text: 'Light Activity', bars: 2 },
    moderate: { color: 'bg-yellow-500', text: 'Moderate Activity', bars: 3 },
    intense: { color: 'bg-red-500', text: 'Intense Activity', bars: 4 }
  };

  const current = levels[level];

  return (
    <div className="bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-700">
      <h3 className="text-lg font-semibold text-white mb-4">Activity Level</h3>
      <div className="flex flex-col items-center gap-4">
        <div className="flex items-end gap-2 h-24">
          {[1, 2, 3, 4].map((bar) => (
            <div
              key={bar}
              className={`w-10 rounded-t-lg transition-all duration-500 ${
                bar <= current.bars ? current.color : 'bg-gray-700'
              }`}
              style={{ height: `${bar * 25}%` }}
            />
          ))}
        </div>
        <div className="text-center">
          <p className="text-xl font-bold text-white">
            {current.text}
          </p>
        </div>
      </div>
    </div>
  );
}

