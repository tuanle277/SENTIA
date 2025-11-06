'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';

// --- Configuration ---
const EMA_QUESTIONS = [
  {
    id: 'activity',
    question: 'What are you primarily doing right now?',
    options: ['Working/Studying', 'Commuting', 'Socializing', 'Relaxing', 'Chores'],
  },
  {
    id: 'social',
    question: 'Who are you with?',
    options: ['Alone', 'With Partner/Family', 'With Friends', 'With Coworkers'],
  },
  {
    id: 'location',
    question: 'How would you describe your environment?',
    options: ['Home', 'Work/School', 'Public Space', 'Outdoors'],
  },
];

// --- Mock Backend/Model Logic ---
const getSuggestionFromModel = (context: Record<string, string>) => {
  const { activity, social, location } = context;
  if (
    activity === 'Working/Studying' &&
    (location === 'Work/School' || location === 'Home')
  ) {
    return social === 'Alone'
      ? {
          title: 'Discreet Focus Reset',
          suggestion:
            'Place your hand flat on your desk. Focus on the cool, solid feeling for 60 seconds. This simple grounding technique can pull you back to the present moment without breaking your workflow.',
        }
      : {
          title: 'Silent Breathing Anchor',
          suggestion:
            'Try a subtle breathing exercise. Inhale slowly for 4 seconds, and exhale for 6. It\'s completely silent and helps calm the nervous system, even if you\'re around others.',
        };
  }
  if (activity === 'Commuting') {
    return {
      title: 'Mindful Commute',
      suggestion:
        'Put on a calming podcast or instrumental playlist. Focus on the sounds and try to loosen your grip if you\'re driving or holding onto a rail. Let the journey be a moment of transition.',
    };
  }
  if (activity === 'Socializing') {
    return {
      title: 'Grounding in the Moment',
      suggestion:
        'If you feel overwhelmed, subtly focus on one thing you can hear in your environment. Let it be an anchor. You don\'t have to leave the conversation, just find a single point of focus.',
    };
  }
  if (activity === 'Relaxing' && location === 'Home') {
    return {
      title: 'Deepen Your Relaxation',
      suggestion:
        'Since you\'re already relaxing at home, enhance it. Try a 5-minute guided meditation or simply listen to one of your favorite, most comforting songs.',
    };
  }
  return {
    title: 'A Mindful Pause',
    suggestion:
      'Take a moment to stand up, stretch your arms to the sky, and take one deep, intentional breath. A small reset can make a big difference.',
  };
};

type AppState =
  | 'IDLE'
  | 'STRESS_DETECTED'
  | 'EMA_START'
  | 'EMA_QUESTIONING'
  | 'FETCHING_SUGGESTION'
  | 'SHOWING_SUGGESTION';

interface Notification {
  id: string;
  type: 'stress_detected' | 'info' | 'warning';
  message: string;
  timestamp: number;
}

interface SystemStatus {
  status: 'running' | 'offline';
  pending_notifications: number;
}

interface Suggestion {
  title: string;
  suggestion: string;
}

interface CompanionAppProps {
  embedded?: boolean;
  onClose?: () => void;
}

export default function CompanionApp({ embedded = false, onClose }: CompanionAppProps) {
  const [appState, setAppState] = useState<AppState>('IDLE');
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [emaAnswers, setEmaAnswers] = useState<Record<string, string>>({});
  const [suggestion, setSuggestion] = useState<Suggestion | null>(null);
  const [isFading, setIsFading] = useState(false);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [stressDetected, setStressDetected] = useState(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);

  const transitionToState = (newState: AppState) => {
    setIsFading(true);
    setTimeout(() => {
      setAppState(newState);
      setIsFading(false);
    }, 300);
  };

  // --- API Functions ---
  const fetchNotifications = async () => {
    try {
      const response = await fetch('/api/notifications');
      const data = await response.json();
      setNotifications(data.notifications || []);

      // Check for stress notifications
      const stressNotifications = data.notifications.filter(
        (n: Notification) => n.type === 'stress_detected'
      );
      if (stressNotifications.length > 0 && !stressDetected) {
        setStressDetected(true);
        // Auto-trigger EMA if stress is detected
        transitionToState('STRESS_DETECTED');
      }
    } catch (error) {
      console.error('Failed to fetch notifications:', error);
    }
  };

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('/api/status');
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Failed to fetch system status:', error);
    }
  };

  const acknowledgeNotification = async (notificationId: string) => {
    try {
      await fetch('/api/notifications', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'acknowledge', id: notificationId }),
      });
      await fetchNotifications(); // Refresh notifications
    } catch (error) {
      console.error('Failed to acknowledge notification:', error);
    }
  };

  const clearAllNotifications = async () => {
    try {
      await fetch('/api/notifications', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'clear' }),
      });
      setNotifications([]);
      setStressDetected(false);
    } catch (error) {
      console.error('Failed to clear notifications:', error);
    }
  };

  // --- Effects ---
  useEffect(() => {
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }

    // Fetch initial data
    fetchNotifications();
    fetchSystemStatus();

    // Set up polling for notifications
    const interval = setInterval(() => {
      fetchNotifications();
      fetchSystemStatus();
    }, 5000); // Check every 5 seconds

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (appState === 'FETCHING_SUGGESTION') {
      setTimeout(() => {
        const result = getSuggestionFromModel(emaAnswers);
        setSuggestion(result);
        transitionToState('SHOWING_SUGGESTION');
      }, 1500);
    }
  }, [appState, emaAnswers]);

  const handleNotificationClick = () => transitionToState('EMA_START');
  const handleStartEma = () => {
    setCurrentQuestionIndex(0);
    setEmaAnswers({});
    transitionToState('EMA_QUESTIONING');
  };

  const handleStressDetected = () => {
    // Clear stress notifications and start EMA
    clearAllNotifications();
    transitionToState('EMA_START');
  };

  const handleAnswer = (questionId: string, answer: string) => {
    const newAnswers = { ...emaAnswers, [questionId]: answer };
    setEmaAnswers(newAnswers);

    if (currentQuestionIndex < EMA_QUESTIONS.length - 1) {
      setIsFading(true);
      setTimeout(() => {
        setCurrentQuestionIndex(currentQuestionIndex + 1);
        setIsFading(false);
      }, 300);
    } else {
      transitionToState('FETCHING_SUGGESTION');
    }
  };

  const handleSkip = () => transitionToState('FETCHING_SUGGESTION');
  const handleReset = () => {
    transitionToState('IDLE');
    setCurrentQuestionIndex(0);
    setEmaAnswers({});
    setSuggestion(null);
  };

  const simulateStressNotification = async () => {
    try {
      await fetch('/api/notifications', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'simulate_stress' }),
      });
      await fetchNotifications();
    } catch (error) {
      console.error('Failed to simulate stress notification:', error);
    }
  };

  const renderContent = () => {
    switch (appState) {
      case 'STRESS_DETECTED':
        return (
          <Card>
            <h1 className="text-2xl font-bold text-red-600 text-center mb-3 mt-0">
              🚨 Stress Detected
            </h1>
            <p className="text-slate-600 text-center mb-7 leading-relaxed max-w-[90%]">
              Our sensors have detected elevated stress levels. Let's take a moment to check in and
              find the best way to help you.
            </p>
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-5">
              <p className="text-sm text-red-900 text-center m-0 leading-relaxed">
                Your physiological indicators suggest you might be experiencing stress. This is
                completely normal, and we're here to help.
              </p>
            </div>
            <Button title="Let's Check In" onClick={handleStressDetected} />
            <button
              onClick={handleReset}
              className="bg-transparent border-none text-slate-400 text-sm font-medium text-center cursor-pointer mt-4 px-2 py-2 hover:text-slate-600 transition-colors"
            >
              Dismiss
            </button>
          </Card>
        );
      case 'EMA_START':
        return (
          <Card>
            <h1 className="text-xl font-bold text-slate-900 text-center mb-3 mt-0">
              Checking In
            </h1>
            <p className="text-slate-600 text-center mb-7 leading-relaxed">
              Let's find the best way to help. A few quick questions will find a personalized
              suggestion for you.
            </p>
            <Button title="Begin" onClick={handleStartEma} />
          </Card>
        );
      case 'EMA_QUESTIONING':
        const question = EMA_QUESTIONS[currentQuestionIndex];
        return (
          <Card key={currentQuestionIndex}>
            <h2 className="text-xl font-bold text-slate-900 text-center mb-7 mt-0">
              {question.question}
            </h2>
            <div className="w-full">
              {question.options.map((option, index) => (
                <Button
                  key={option}
                  title={option}
                  onClick={() => handleAnswer(question.id, option)}
                  className="animate-in fade-in"
                  style={{ animationDelay: `${index * 100}ms` }}
                />
              ))}
            </div>
            <button
              onClick={handleSkip}
              className="bg-transparent border-none text-slate-400 text-sm font-medium text-center cursor-pointer mt-4 px-2 py-2 hover:text-slate-600 transition-colors"
            >
              Skip
            </button>
          </Card>
        );
      case 'FETCHING_SUGGESTION':
        return (
          <Card>
            <p className="text-slate-600 text-center mb-5">Analyzing your context...</p>
            <div className="border-4 border-slate-200 border-t-slate-600 rounded-full w-9 h-9 animate-spin mx-auto my-5" />
          </Card>
        );
      case 'SHOWING_SUGGESTION':
        return (
          <Card>
            <h1 className="text-xl font-bold text-slate-900 text-center mb-3 mt-0">
              {suggestion?.title}
            </h1>
            <p className="text-base text-slate-700 text-center mb-7 leading-relaxed mt-0">
              {suggestion?.suggestion}
            </p>
            <Button title="Done" onClick={handleReset} />
          </Card>
        );
      case 'IDLE':
      default:
        return (
          <Card>
            <h1 className="text-xl font-bold text-slate-900 text-center mb-3 mt-0">
              Mindwell Companion
            </h1>
            <p className="text-slate-600 text-center mb-7 leading-relaxed">
              Your personal guide to moments of calm, triggered by your wearable when you need it
              most.
            </p>

            {/* System Status */}
            {systemStatus && (
              <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 mb-4">
                <p className="text-sm text-slate-600 m-0 text-center">
                  Status:{' '}
                  {systemStatus.status === 'running' ? '🟢 Monitoring' : '🔴 Offline'}
                </p>
                {systemStatus.pending_notifications > 0 && (
                  <p className="text-xs text-red-600 m-1 mt-0 text-center font-medium">
                    {systemStatus.pending_notifications} notification(s) pending
                  </p>
                )}
              </div>
            )}

            {/* Notifications */}
            {notifications.length > 0 && (
              <div className="bg-amber-50 border border-amber-300 rounded-xl p-4 mb-4">
                <h3 className="text-base font-semibold text-amber-900 m-0 mb-3 text-center">
                  Recent Alerts
                </h3>
                {notifications.slice(0, 3).map((notification) => (
                  <div
                    key={notification.id}
                    className="bg-white border border-slate-200 rounded-lg p-3 mb-2 flex justify-between items-center last:mb-0"
                  >
                    <p className="text-xs text-slate-700 m-0 flex-1 leading-snug">
                      {notification.message}
                    </p>
                    <button
                      onClick={() => acknowledgeNotification(notification.id)}
                      className="bg-slate-500 text-white border-none rounded-md px-2 py-1 text-xs cursor-pointer ml-2"
                    >
                      Dismiss
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div className="mt-5" />
            <Button title="Simulate Stress Notification" onClick={simulateStressNotification} />
            {notifications.length > 0 && (
              <Button
                title="Clear All Notifications"
                onClick={clearAllNotifications}
                className="bg-red-600 mt-2"
              />
            )}
          </Card>
        );
    }
  };

  if (embedded) {
    return (
      <div className="flex justify-center items-center bg-slate-100 p-4 box-border relative min-h-full w-full">
        <div className={`transition-opacity duration-300 w-full ${isFading ? 'opacity-0' : 'opacity-100'}`}>
          {renderContent()}
        </div>
      </div>
    );
  }

  return (
    <main className="flex min-h-screen justify-center items-center bg-slate-100 p-5 box-border relative">
      <Link
        href="/"
        className="absolute top-8 left-8 flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors text-sm font-medium"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Dashboard
      </Link>
      <div className={`transition-opacity duration-300 ${isFading ? 'opacity-0' : 'opacity-100'}`}>
        {renderContent()}
      </div>
    </main>
  );
}

// --- Reusable Components ---
const Card = ({ children }: { children: React.ReactNode }) => (
  <div className="w-full max-w-[380px] bg-white rounded-3xl p-8 flex flex-col items-center justify-center border border-slate-200 shadow-sm box-border mx-auto">
    {children}
  </div>
);

const Button = ({
  title,
  onClick,
  className = '',
  style,
}: {
  title: string;
  onClick: () => void;
  className?: string;
  style?: React.CSSProperties;
}) => (
  <button
    className={`w-full bg-slate-900 py-4 rounded-xl mt-2.5 text-white text-center text-sm font-medium border-none cursor-pointer shadow-sm transition-all duration-200 hover:bg-slate-800 hover:-translate-y-0.5 active:translate-y-0 ${className}`}
    onClick={onClick}
    style={style}
  >
    {title}
  </button>
);

