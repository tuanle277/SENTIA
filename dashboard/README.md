# Health Dashboard

A real-time health monitoring dashboard that displays live metrics from wearable devices. Built with Next.js 15, TypeScript, and Tailwind CSS.

## Features

- **Real-time Data Streaming**: Server-Sent Events (SSE) for live metric updates
- **Comprehensive Metrics**:
  - Heart Rate monitoring with status indicators
  - Stress Level tracking
  - Blood Oxygen (SpO2) levels
  - Body Temperature
  - Steps and Calories burned
  - Sleep Quality scores
  - Activity Level visualization
- **Interactive Charts**: Real-time heart rate and stress level trends
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Status Indicators**: Visual alerts for abnormal readings
- **Connection Monitoring**: Live connection status with auto-reconnect

## Tech Stack

- **Framework**: Next.js 15 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS 4
- **Charts**: Recharts
- **Icons**: Lucide React
- **Data Streaming**: Server-Sent Events (SSE)

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Navigate to the dashboard directory:
```bash
cd dashboard
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

### Production Build

```bash
npm run build
npm start
```

## Project Structure

```
dashboard/
├── app/
│   ├── api/
│   │   └── stream/
│   │       └── route.ts          # SSE endpoint for data streaming
│   ├── components/
│   │   ├── MetricCard.tsx        # Individual metric display
│   │   ├── RealtimeChart.tsx     # Line/area charts
│   │   └── ActivityIndicator.tsx # Activity level bars
│   ├── page.tsx                  # Main dashboard page
│   ├── layout.tsx                # Root layout
│   └── globals.css               # Global styles
├── public/                       # Static assets
└── README.md                     # This file
```

## Data Simulation

The dashboard includes two ways to simulate wearable data:

### 1. Built-in SSE Stream (Default)
The Next.js API route at `/api/stream` automatically generates and streams realistic wearable data. This is used by default when you run the dashboard.


## Data Format

The wearable data follows this format:

```typescript
interface WearableData {
  timestamp: number;           // Unix timestamp in milliseconds
  heartRate: number;           // Beats per minute (50-180)
  stressLevel: number;         // Percentage (0-100)
  spo2: number;               // Blood oxygen percentage (90-100)
  temperature: number;         // Body temperature in Celsius (35-38)
  steps: number;              // Cumulative daily steps
  calories: number;           // Cumulative calories burned
  sleepQuality: number;       // Sleep quality score (0-100)
  activityLevel: string;      // 'resting' | 'light' | 'moderate' | 'intense'
}
```

## Customization

### Adjusting Data Generation
Edit `app/api/stream/route.ts` to modify:
- Update interval (currently 1 second)
- Value ranges and variations
- Number of data points kept in history

### Styling
The dashboard uses Tailwind CSS. Customize colors and styles in:
- `tailwind.config.js` - Theme configuration
- `app/globals.css` - Global styles
- Component files - Component-specific styles

### Adding New Metrics
1. Add the metric to the `WearableData` interface
2. Update the data generation in `app/api/stream/route.ts`
3. Create or update components to display the new metric
4. Add the metric to the dashboard layout in `app/page.tsx`

## Health Status Indicators

The dashboard automatically determines health status based on these thresholds:

- **Heart Rate**:
  - Normal: 60-100 BPM
  - Warning: <60 or >100 BPM
  - Danger: >120 BPM

- **Stress Level**:
  - Normal: <50%
  - Warning: 50-70%
  - Danger: >70%

- **Blood Oxygen (SpO2)**:
  - Normal: ≥95%
  - Warning: 90-95%
  - Danger: <90%

- **Temperature**:
  - Normal: 36-37.5°C
  - Warning: <36 or 37.5-38°C
  - Danger: >38°C

## Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Performance

- Updates every second with minimal CPU usage
- Maintains 60 data points (1 minute) of history
- Optimized re-renders using React hooks
- Smooth animations without jank

## Troubleshooting

### Connection Issues
If the dashboard shows "disconnected":
1. Ensure the Next.js dev server is running
2. Check browser console for errors
3. The dashboard will auto-reconnect after 3 seconds

### No Data Displayed
1. Check that `/api/stream` endpoint is accessible
2. Verify browser supports EventSource API
3. Check for any CORS issues in browser console

### Performance Issues
1. Reduce the number of data points kept in history
2. Increase the update interval
3. Disable animations in component props

## Future Enhancements

Potential additions:
- Historical data storage and playback
- Multiple user profiles
- Alert notifications for abnormal readings
- Data export functionality
- Integration with real wearable APIs
- Machine learning predictions
- Comparison views
- Custom dashboard layouts

## License

MIT

## Support

For issues or questions, please create an issue in the repository.
